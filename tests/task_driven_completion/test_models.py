"""Synthetic CPU numerical, gradient, calibration and frozen-backbone checks."""
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from inm.encoder_candidates.transformer import SpatialTransformer
from inm.supervised_tucker.models import observed_ridge_core
from inm.task_driven_completion.adapters import CompletionAdapter
from inm.task_driven_completion.completion import (
    CovarianceCompleter, TuckerCompleter, completion_state_arrays,
    fit_initialization, paired_completers, reconstruction_loss,
    restore_completer, scoped_cpu_rng, validate_input,
)


def fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(349)
    raw = torch.randn(2, 22, 4, 250, generator=generator, dtype=dtype)
    u = torch.randn(22, 4, generator=generator, dtype=dtype)
    v = torch.randn(250, 16, generator=generator, dtype=dtype)
    u, v = u / u.norm(dim=0), v / v.norm(dim=0)
    mixing = torch.randn(22, 22, generator=generator, dtype=torch.float64)
    moment = mixing @ mixing.T / 22 + torch.eye(22, dtype=torch.float64) * 0.1
    mask = torch.ones(2, 22, 4, dtype=torch.bool)
    mask[0, 6:, 0] = False
    mask[0, 16:, 1:] = False
    mask[1, :6, :] = False
    return raw, mask, {"U": u, "V": v, "second_moment": moment,
                       "training_trials": torch.tensor(2)}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.raw, self.mask, self.initial = fixture()

    def test_observed_design_reference_and_normalized_ridge(self):
        # Use all 250 samples: penalty must be n_observed * 250 * .001.
        raw, mask, initial = fixture(torch.float64)
        mask[0, :, 0] = False
        mask[0, :2, 0] = True  # Fewer electrodes than channel rank.
        u, v = initial["U"], initial["V"]
        core = observed_ridge_core(raw, mask, u, v)
        for batch, window in ((0, 0), (0, 1), (1, 3)):
            known = mask[batch, :, window]
            design = torch.einsum("cr,ts->ctrs", u[known], v).reshape(-1, 64)
            target = raw[batch, known, window].reshape(-1)
            reference = torch.linalg.solve(
                design.T @ design + known.sum() * 250 * .001 * torch.eye(64, dtype=torch.float64),
                design.T @ target).reshape(4, 16)
            torch.testing.assert_close(core[batch, :, window], reference, rtol=1e-11, atol=1e-12)
        # Repeated columns remain solvable under strictly positive ridge.
        repeated = u[:, :1].expand(-1, 4)
        self.assertTrue(torch.isfinite(observed_ridge_core(raw, mask, repeated, v)).all())

    def test_factor_gradient_matches_central_finite_difference(self):
        raw, mask, initial = fixture(torch.float64)
        model = paired_completers(initial)["tucker_learned"]
        weights = torch.linspace(-.7, .9, raw.numel(), dtype=raw.dtype).reshape_as(raw)
        def objective():
            return (model(raw, mask) * weights).sum() / 100
        objective().backward()
        for parameter, index in ((model.U, (10, 2)), (model.V, (7, 3))):
            analytic = parameter.grad[index].item()
            epsilon = 1e-5
            with torch.no_grad():
                original = parameter[index].item()
                parameter[index] = original + epsilon
                plus = objective().item()
                parameter[index] = original - epsilon
                minus = objective().item()
                parameter[index] = original
            numeric = (plus - minus) / (2 * epsilon)
            self.assertGreater(abs(analytic), 1e-8)
            self.assertAlmostEqual(analytic, numeric, delta=1e-7)

    def test_covariance_reconstructs_stabilized_moment_and_conditional_reference(self):
        raw, mask, initial = fixture(torch.float64)
        model = CovarianceCompleter(initial["second_moment"], learned=True)
        sigma = initial["second_moment"] + 1e-6 * torch.eye(22, dtype=torch.float64)
        torch.testing.assert_close(model.covariance(), sigma, rtol=1e-12, atol=1e-12)
        actual = model(raw, mask)
        for batch, window in ((0, 0), (0, 1), (1, 3)):
            known, missing = mask[batch, :, window], ~mask[batch, :, window]
            system = sigma[known][:, known] + .001 * sigma.diag().mean() * torch.eye(int(known.sum()), dtype=torch.float64)
            reference = sigma[missing][:, known] @ torch.linalg.solve(system, raw[batch, known, window])
            torch.testing.assert_close(actual[batch, missing, window], reference, rtol=1e-11, atol=1e-12)

    def test_covariance_gradient_matches_finite_difference(self):
        raw, mask, initial = fixture(torch.float64)
        model = CovarianceCompleter(initial["second_moment"], learned=True)
        def objective():
            return model(raw, mask)[~mask].square().mean()
        objective().backward()
        # Include diagonal and off-diagonal free Cholesky coordinates.
        for index in (0, 17, 252):
            analytic = model.free.grad[index].item()
            epsilon = 1e-5
            with torch.no_grad():
                original = model.free[index].item()
                model.free[index] = original + epsilon
                plus = objective().item()
                model.free[index] = original - epsilon
                minus = objective().item()
                model.free[index] = original
            self.assertGreater(abs(analytic), 1e-8)
            self.assertAlmostEqual(analytic, (plus - minus) / (2 * epsilon), delta=1e-7)

    def test_paired_initial_states_and_inference_are_bitwise_identical(self):
        models = paired_completers(self.initial)
        for family in ("tucker", "covariance"):
            frozen, learned = models[family + "_frozen"], models[family + "_learned"]
            self.assertFalse(list(frozen.parameters()))
            self.assertTrue(all(parameter.requires_grad for parameter in learned.parameters()))
            for name, value in frozen.state_dict().items():
                self.assertTrue(torch.equal(value, learned.state_dict()[name]))
                self.assertNotEqual(value.data_ptr(), learned.state_dict()[name].data_ptr())
            self.assertTrue(torch.equal(frozen(self.raw, self.mask), learned(self.raw, self.mask)))
            self.assertEqual(learned.anchor_penalty().item(), 0)

    def test_hidden_nan_inf_invariance_and_bitwise_observation_preservation(self):
        for model in paired_completers(self.initial).values():
            expected = model(self.raw, self.mask)
            self.assertTrue(torch.equal(expected[self.mask], self.raw[self.mask]))
            for placeholder in (float("nan"), float("inf"), -float("inf"), 12345.):
                changed = self.raw.clone()
                changed[~self.mask] = placeholder
                self.assertTrue(torch.equal(model(changed, self.mask), expected))

    def test_hidden_input_gradients_are_zero_even_for_nan_placeholders(self):
        for strategy in ("tucker_learned", "covariance_learned"):
            raw = self.raw.clone()
            raw[~self.mask] = float("nan")
            raw.requires_grad_(True)
            model = paired_completers(self.initial)[strategy]
            model(raw, self.mask).square().mean().backward()
            self.assertTrue(torch.isfinite(raw.grad).all())
            self.assertTrue(torch.equal(raw.grad[~self.mask], torch.zeros_like(raw.grad[~self.mask])))
            self.assertGreater(raw.grad[self.mask].abs().sum().item(), 0)

    def test_completion_is_window_local(self):
        changed = self.raw.clone()
        changed[:, :, 1:] += 100
        for model in paired_completers(self.initial).values():
            expected = model(self.raw, self.mask)
            actual = model(changed, self.mask)
            self.assertTrue(torch.equal(expected[:, :, 0], actual[:, :, 0]))

    def test_full_masks_bypass_solve_and_have_no_completion_gradient(self):
        full = torch.ones_like(self.mask)
        raw = self.raw.clone().requires_grad_(True)
        for model in paired_completers(self.initial).values():
            result = model(raw, full)
            self.assertIs(result, raw)
            parameters = tuple(model.parameters())
            if parameters:
                grads = torch.autograd.grad(result.sum(), parameters, allow_unused=True)
                self.assertTrue(all(gradient is None for gradient in grads))

    def test_mixed_batch_full_trials_have_zero_factor_gradient(self):
        mask = self.mask.clone()
        mask[0] = True
        for strategy in ("tucker_learned", "covariance_learned"):
            model = paired_completers(self.initial)[strategy]
            result = model(self.raw, mask)
            self.assertTrue(torch.equal(result[0], self.raw[0]))
            result[0].sum().backward()
            for parameter in model.parameters():
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

    def test_rejects_malformed_masks_observed_nonfinite_and_empty_inputs(self):
        empty = self.mask.clone()
        empty[0, :, 2] = False
        observed_nan = self.raw.clone()
        observed_nan[0, 0, 0, 0] = float("nan")
        for raw, mask in ((self.raw, self.mask.float()), (self.raw, self.mask[:, :, :3]),
                          (self.raw, empty), (observed_nan, self.mask),
                          (self.raw[:0], self.mask[:0]), (self.raw[..., :249], self.mask)):
            with self.assertRaises(ValueError):
                validate_input(raw, mask)

    def test_anchor_displacement_and_post_step_constraints(self):
        models = paired_completers(self.initial)
        tucker, covariance = models["tucker_learned"], models["covariance_learned"]
        with torch.no_grad():
            tucker.U.add_(.1)
            tucker.V.add_(.2)
            covariance.free.add_(.3)
        expected = (.01 * 88 + .04 * 4000) / (88 + 4000)
        self.assertAlmostEqual(tucker.anchor_penalty().item(), expected, places=7)
        self.assertAlmostEqual(covariance.anchor_penalty().item(), .09, places=12)
        for model in (tucker, covariance):
            model.anchor_penalty().backward()
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
            model.post_step()
        for factor in (tucker.U, tucker.V):
            torch.testing.assert_close(factor.norm(dim=0), torch.ones(factor.shape[1]))
        self.assertTrue((torch.linalg.eigvalsh(covariance.covariance()) > 0).all())

    def test_reconstruction_loss_uses_only_hidden_targets_and_empty_zero_is_differentiable(self):
        completed = self.raw.clone().requires_grad_(True)
        target = self.raw.clone()
        target[~self.mask] += 2
        loss = reconstruction_loss(completed, target, self.mask)
        self.assertEqual(loss.item(), 4.)
        loss.backward()
        self.assertTrue(torch.equal(completed.grad[self.mask], torch.zeros_like(completed.grad[self.mask])))
        full = torch.ones_like(self.mask)
        completed.grad = None
        zero = reconstruction_loss(completed, target, full)
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertTrue(torch.equal(completed.grad, torch.zeros_like(completed.grad)))
        zero_without_input_graph = reconstruction_loss(completed.detach(), target, full)
        self.assertEqual(zero_without_input_graph.item(), 0.)
        self.assertTrue(zero_without_input_graph.requires_grad)
        zero_without_input_graph.backward()

    def test_numeric_state_roundtrip_after_update_and_strict_rejection(self):
        for strategy, model in paired_completers(self.initial).items():
            if strategy.endswith("learned"):
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                model(self.raw, self.mask)[~self.mask].square().mean().backward()
                optimizer.step()
                model.post_step()
            arrays = completion_state_arrays(model)
            restored = restore_completer(strategy, arrays)
            self.assertTrue(torch.equal(model(self.raw, self.mask), restored(self.raw, self.mask)))
            self.assertEqual(model.anchor_penalty().item(), restored.anchor_penalty().item())
            with self.assertRaises(ValueError):
                restore_completer(strategy, {**arrays, "foreign": np.array(0.)})
            if arrays:
                bad = {name: array.copy() for name, array in arrays.items()}
                next(iter(bad.values())).flat[0] = np.nan
                with self.assertRaises(ValueError):
                    restore_completer(strategy, bad)

    def test_calibration_fixed_settings_train_only_and_rng_restore(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.get_num_threads())
        initial, history = fit_initialization(self.raw, 2)
        self.assertEqual(len(history), 30)
        self.assertEqual(initial["training_trials"].item(), len(self.raw))
        rows = self.raw.permute(0, 2, 3, 1).reshape(-1, 22).double()
        torch.testing.assert_close(initial["second_moment"], rows.T @ rows / len(rows), rtol=1e-12, atol=1e-12)
        self.assertEqual(random.getstate(), before[0])
        self.assertEqual(np.random.get_state()[0], before[1][0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(torch.get_num_threads(), before[3])
        # Change ambient streams and check exact repeat calibration.
        with scoped_cpu_rng(932):
            repeated, repeated_history = fit_initialization(self.raw, 2)
        for name in initial:
            self.assertTrue(torch.equal(initial[name], repeated[name]), name)
        self.assertEqual(history, repeated_history)
        paired_completers(initial)
        with self.assertRaisesRegex(ValueError, "train partition"):
            fit_initialization(self.raw, 2, partition="validation")

    def test_rng_scope_restores_all_states_on_exception(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.get_num_threads())
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with scoped_cpu_rng(93, threads=2):
                random.random(); np.random.rand(); torch.rand(3)
                raise RuntimeError("synthetic failure")
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_equal(np.random.get_state(), before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(torch.get_num_threads(), before[3])

    def test_calibration_restores_rng_on_fitter_failure(self):
        before = torch.get_rng_state().clone()
        def fail(*args, **kwargs):
            torch.rand(12)
            raise RuntimeError("fit failed")
        with patch("inm.task_driven_completion.completion.Tucker2.fit", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                fit_initialization(self.raw, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_adapter_exact_zero_full_paths_and_original_flags(self):
        model = SpatialTransformer(seed=11).eval()
        full = torch.ones_like(self.mask)
        for strategy, completer in paired_completers(self.initial).items():
            adapter = CompletionAdapter(model, strategy, completer)
            with patch.object(completer, "complete", side_effect=AssertionError("full bypass")):
                self.assertTrue(torch.equal(adapter(self.raw, full), model(self.raw, full)))
            if strategy == "zero":
                self.assertTrue(torch.equal(adapter(self.raw, self.mask), model(self.raw, self.mask)))
            captured = []
            handle = model.classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
            adapter(self.raw, self.mask)
            handle.remove()
            self.assertTrue(torch.equal(captured[0][:, -88:], self.mask.reshape(2, -1).float()))

    def test_classification_gradients_train_completion_through_frozen_eval_backbone(self):
        for strategy in ("tucker_learned", "covariance_learned"):
            backbone = SpatialTransformer(seed=31)
            before = {name: value.clone() for name, value in backbone.state_dict().items()}
            model = paired_completers(self.initial)[strategy]
            adapter = CompletionAdapter(backbone, strategy, model).train()
            self.assertTrue(adapter.training)
            self.assertTrue(model.training)
            self.assertTrue(all(not module.training for module in backbone.modules()))
            raw = self.raw.clone().requires_grad_(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            initial = {name: value.clone() for name, value in model.named_parameters()}
            # Classification alone proves that completion is not detached.
            F.cross_entropy(adapter(raw, self.mask), torch.tensor([0, 3])).backward()
            self.assertGreater(raw.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(raw.grad).all())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()
            model.post_step()
            self.assertTrue(all(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
            for name, value in backbone.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)

    def test_adapter_hidden_placeholders_and_no_full_classification_factor_gradient(self):
        for strategy in ("tucker_learned", "covariance_learned"):
            completer = paired_completers(self.initial)[strategy]
            adapter = CompletionAdapter(SpatialTransformer(seed=9), strategy, completer).train()
            hidden = self.raw.clone()
            hidden[~self.mask] = float("nan")
            self.assertTrue(torch.equal(adapter(hidden, self.mask), adapter(self.raw, self.mask)))
            raw = self.raw.clone().requires_grad_(True)
            full = torch.ones_like(self.mask)
            F.cross_entropy(adapter(raw, full), torch.tensor([0, 1])).backward()
            self.assertTrue(all(parameter.grad is None for parameter in completer.parameters()))
            self.assertGreater(raw.grad.abs().sum().item(), 0)

    def test_adapter_rejects_external_backbone_training_or_grad_changes(self):
        backbone = SpatialTransformer(seed=8)
        adapter = CompletionAdapter(backbone)
        backbone.temporal_bn.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            adapter(self.raw, self.mask)
        adapter.train()
        self.assertFalse(backbone.temporal_bn.training)
        backbone.temporal_conv.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "must not require"):
            adapter(self.raw, self.mask)

    def test_adapter_rejects_strategy_family_or_learning_mismatch(self):
        backbone = SpatialTransformer(seed=8)
        models = paired_completers(self.initial)
        for strategy, mismatched in (("covariance_frozen", "tucker_frozen"),
                                     ("covariance_frozen", "covariance_learned"),
                                     ("tucker_learned", "tucker_frozen")):
            with self.assertRaisesRegex(ValueError, "must match"):
                CompletionAdapter(backbone, strategy, models[mismatched])


if __name__ == "__main__":
    unittest.main()
