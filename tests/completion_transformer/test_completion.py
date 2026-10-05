import random
import tempfile
import unittest

import numpy as np
import torch

from inm.completion_transformer import (
    CovarianceCompleter, TuckerCompleter,
)
from inm.completion_transformer.completion import completion_state_arrays, load_completion_state


class CompletionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(11)
        self.train = torch.randn(8, 3, 4, 4, generator=generator)
        self.x = torch.randn(2, 3, 4, 4, generator=generator)
        self.mask = torch.ones(2, 3, 4, dtype=torch.bool)
        self.mask[:, 1, :] = False

    def test_covariance_observed_preservation_hidden_nonfinite_and_reload(self):
        model = CovarianceCompleter(channels=3).fit(self.train)
        x = self.x.clone()
        x[:, 1] = float("nan")
        completed = model.complete(x, self.mask)
        self.assertTrue(torch.isfinite(completed).all())
        self.assertTrue(torch.equal(completed[self.mask], x[self.mask]))
        clone = CovarianceCompleter(channels=3)
        clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(completed, clone.complete(x, self.mask)))
        x[:, 1] = float("inf")
        self.assertTrue(torch.equal(completed, model.complete(x, self.mask)))

    def test_covariance_matches_explicit_conditional_ridge_oracle(self):
        model = CovarianceCompleter(channels=3).fit(self.train)
        sample = self.x[:1]
        mask = torch.tensor([[[True] * 4, [False] * 4, [True] * 4]])
        actual = model.complete(sample, mask)
        sigma = model.second_moment.double()
        known = torch.tensor([0, 2])
        missing = torch.tensor([1])
        ridge = model.ridge_scale * sigma.diag().mean()
        beta = torch.linalg.solve(
            sigma[known][:, known] + ridge * torch.eye(len(known), dtype=torch.float64),
            sigma[known][:, missing])
        expected_missing = beta.T @ sample[0, known, 0].double()
        torch.testing.assert_close(actual[0, missing, 0].double(), expected_missing, rtol=2e-5, atol=2e-6)

    def test_tucker_ridge_matches_explicit_observed_entry_oracle(self):
        completer = TuckerCompleter(channels=3, windows=4, samples=4,
                                    rank_channels=2, rank_features=3,
                                    epochs=2, batch_size=4)
        completer.fit(self.train, task_seed=7)
        tucker = completer.model
        x = self.x[:1].clone()
        mask = torch.tensor([[[True] * 4, [False] * 4, [True] * 4]])
        safe = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        actual = tucker._solve(safe, mask, tucker.U, tucker.V)
        expected = torch.empty_like(actual)
        for window in range(4):
            channels = torch.nonzero(mask[0, :, window], as_tuple=False).flatten()
            u = tucker.U[channels]
            v = tucker.V
            design = torch.einsum("cr,fs->cfrs", u, v).reshape(-1, u.shape[1] * v.shape[1])
            target = safe[0, channels, window].reshape(-1)
            penalty = len(channels) * tucker.features * tucker.ridge
            gram = design.T @ design + penalty * torch.eye(design.shape[1])
            rhs = design.T @ target
            core = torch.linalg.solve(gram, rhs).reshape(tucker.rank_channels, tucker.rank_features)
            expected[0, :, window, :] = core
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_tucker_rng_scope_reload_and_hidden_nonfinite(self):
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        completer = TuckerCompleter(channels=3, windows=4, samples=4,
                                    rank_channels=2, rank_features=3,
                                    epochs=2, batch_size=4)
        completer.fit(self.train, task_seed=31)
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        x = self.x.clone()
        x[:, 1] = float("nan")
        result = completer.complete(x, self.mask)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[self.mask], x[self.mask]))
        x[:, 1] = float("inf")
        self.assertTrue(torch.equal(result, completer.complete(x, self.mask)))
        clone = TuckerCompleter(channels=3, windows=4, samples=4,
                                rank_channels=2, rank_features=3, epochs=2, batch_size=4)
        clone.load_state_dict(completer.state_dict())
        self.assertTrue(torch.equal(result, clone.complete(x, self.mask)))

    def test_tucker_fit_repeatable_across_ambient_rng_and_restores_on_error(self):
        settings = dict(channels=3, windows=4, samples=4, rank_channels=2,
                        rank_features=3, epochs=2, batch_size=4)
        first = TuckerCompleter(**settings)
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        first.fit(self.train, task_seed=53)
        first_state = {name: value.clone() for name, value in first.state_dict().items()}

        second = TuckerCompleter(**settings)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        second.fit(self.train, task_seed=53)
        for name, value in second.state_dict().items():
            self.assertTrue(torch.equal(value, first_state[name]), name)

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        completer = TuckerCompleter(**settings)

        def fail_after_random_draw(*_args, **_kwargs):
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("synthetic fit failure")

        completer.model.fit = fail_after_random_draw
        with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
            completer.fit(self.train, task_seed=61)
        self.assertEqual(py_state, random.getstate())
        restored_np_state = np.random.get_state()
        self.assertEqual(np_state[0], restored_np_state[0])
        self.assertTrue(np.array_equal(np_state[1], restored_np_state[1]))
        self.assertEqual(np_state[2:], restored_np_state[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_primitive_state_round_trip_preserves_completions_and_checks_flags(self):
        covariance = CovarianceCompleter(channels=3).fit(self.train)
        tucker = TuckerCompleter(channels=3, windows=4, samples=4,
            rank_channels=2, rank_features=3, epochs=2, batch_size=4)
        history = tucker.fit(self.train, task_seed=87)
        arrays = completion_state_arrays(covariance, tucker)
        with tempfile.NamedTemporaryFile(suffix=".npz") as stream:
            np.savez_compressed(stream.name, **arrays)
            restored_cov, restored_tucker = load_completion_state(
                stream.name, history, expected_epochs=2)
        torch.testing.assert_close(restored_cov.complete(self.x, self.mask),
                                   covariance.complete(self.x, self.mask), rtol=0, atol=0)
        torch.testing.assert_close(restored_tucker.complete(self.x, self.mask),
                                   tucker.complete(self.x, self.mask), rtol=0, atol=0)

        arrays["tucker__fitted"] = np.asarray(False)
        with tempfile.NamedTemporaryFile(suffix=".npz") as stream:
            np.savez_compressed(stream.name, **arrays)
            with self.assertRaisesRegex(ValueError, "not marked fitted|fitted flag"):
                load_completion_state(stream.name, history, expected_epochs=2)
        arrays["tucker__fitted"] = np.asarray(True)
        arrays["tucker__U"] = np.zeros((3, 9), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz") as stream:
            np.savez_compressed(stream.name, **arrays)
            with self.assertRaisesRegex(ValueError, "invalid dimensions"):
                load_completion_state(stream.name, history, expected_epochs=2)

    def test_all_missing_window_rejected(self):
        mask = self.mask.clone()
        mask[0, :, 0] = False
        with self.assertRaisesRegex(ValueError, "at least one observed"):
            CovarianceCompleter(channels=3).fit(self.train).complete(self.x, mask)

if __name__ == "__main__":
    unittest.main()
