"""Meaningful isolation, masking, tensor-algebra and checkpoint checks."""
import unittest

import torch

from inm.tensor_temporal.models import ARM_IDS, build_model, restore_model


class TensorTemporalModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, count=3):
        generator = torch.Generator().manual_seed(123)
        raw = torch.randn(count, 22, 4, 250, generator=generator)
        mask = torch.ones(count, 22, 4, dtype=torch.bool)
        mask[:, 3:7, 1:3] = False
        return raw, mask

    def calibrated(self, arm):
        model = build_model(arm, seed=2)
        raw, _ = self.inputs()
        report = model.calibrate(raw, tucker_epochs=1)
        model.eval()
        return model, report

    def test_tensor_and_dense_start_with_same_operator_and_head(self):
        raw, mask = self.inputs()
        factored = build_model("learned_tensor_patch", 9).eval()
        dense = build_model("dense_temporal_patch", 9).eval()
        torch.testing.assert_close(factored.U, dense.U, rtol=0, atol=0)
        expanded = torch.einsum("ik,jl->ijkl", factored.A, factored.B).reshape(100, 8)
        torch.testing.assert_close(expanded, dense.V, rtol=2e-6, atol=1e-7)
        for name, value in factored.state_dict().items():
            if name.startswith(("projection.", "transformer.", "classifier.", "mask_projection.")):
                torch.testing.assert_close(value, dense.state_dict()[name], rtol=0, atol=0)
        # Different contraction order can amplify roundoff near log epsilon.
        torch.testing.assert_close(factored(raw, mask), dense(raw, mask), rtol=2e-4, atol=2e-4)
        factored_count = sum(p.numel() for p in factored.parameters())
        dense_count = sum(p.numel() for p in dense.parameters())
        self.assertEqual(dense_count - factored_count, 740)

    def test_hidden_nans_shapes_and_checkpoint_roundtrip_all_arms(self):
        raw, mask = self.inputs()
        hidden = torch.where(mask[..., None], raw, torch.full_like(raw, float("nan")))
        for arm in ARM_IDS:
            with self.subTest(arm=arm):
                model, _ = self.calibrated(arm)
                logits = model(raw, mask)
                self.assertEqual(tuple(logits.shape), (3, 4))
                self.assertTrue(torch.isfinite(logits).all())
                torch.testing.assert_close(logits, model(hidden, mask), rtol=0, atol=0)
                restored = restore_model(arm, model.constructor_settings(), model.state_dict()).eval()
                torch.testing.assert_close(logits, restored(raw, mask), rtol=0, atol=0)

    def test_patch_coverage_and_spectral_locality(self):
        model = build_model("spectral_dense", 0)
        raw, _ = self.inputs()
        patches = model._patches(raw)
        torch.testing.assert_close(patches[:, :, 0], raw[:, :, 0, :100])
        torch.testing.assert_close(patches[:, :, -1], raw[:, :, -1, 150:250])
        changed = raw.clone()
        changed[:, 2, 1] += 100
        before, after = model._spectral(raw), model._spectral(changed)
        untouched = torch.ones_like(before, dtype=torch.bool)
        untouched[:, 2, 4:8] = False
        torch.testing.assert_close(before[untouched], after[untouched], rtol=0, atol=0)

    def test_calibration_statistics_frozen_and_recalibration_rejected(self):
        model, report = self.calibrated("fixed_spectral_tucker")
        raw, _ = self.inputs()
        spectral = model._spectral(raw)
        torch.testing.assert_close(model.spectral_mean, spectral.mean((0, 2), keepdim=True))
        self.assertEqual(len(report["factor_history"]), 1)
        self.assertEqual(report["factor_epochs"], 1)
        self.assertFalse(any(name.startswith("tucker.") for name, _ in model.named_parameters()))
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        model(raw * 20)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            model.calibrate(raw)

    def test_gradients_reach_supervised_factors_and_head(self):
        raw, mask = self.inputs()
        for arm in ARM_IDS[:4]:
            with self.subTest(arm=arm):
                model, _ = self.calibrated(arm)
                loss = torch.nn.functional.cross_entropy(model(raw, mask), torch.tensor([0, 1, 2]))
                loss.backward()
                required = ["classifier.weight"]
                if arm == "learned_tensor_patch":
                    required += ["U", "A", "B"]
                elif arm == "dense_temporal_patch":
                    required += ["U", "V"]
                elif arm == "spectral_dense":
                    required += ["dense_spectral.weight"]
                parameters = dict(model.named_parameters())
                for name in required:
                    gradient = parameters[name].grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.abs().sum()), 0)

    def test_seed_scope_and_validation(self):
        raw, mask = self.inputs()
        torch.manual_seed(999)
        state = torch.random.get_rng_state().clone()
        model = build_model("fixed_spectral_tucker", 5)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        model.calibrate(raw, tucker_epochs=1)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        mask[:, :, 0] = False
        with self.assertRaises(ValueError):
            model(raw, mask)
        with self.assertRaises(ValueError):
            model(raw.double())
        observed_nan = raw.clone()
        observed_nan[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(observed_nan)

    def test_raw_pair_can_overfit_tiny_fixed_batch(self):
        """Optimization sanity only: memorizing eight examples is not EEG evidence."""
        generator = torch.Generator().manual_seed(404)
        raw = torch.randn(8, 22, 4, 250, generator=generator)
        labels = torch.arange(8) % 4
        for arm in ("learned_tensor_patch", "dense_temporal_patch"):
            with self.subTest(arm=arm):
                # Disable dropout to make a deterministic optimizer check.
                model = build_model(arm, 3).eval()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)
                initial = torch.nn.functional.cross_entropy(model(raw), labels).detach()
                for _ in range(30):
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(raw), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    model.clip_weights()
                final = torch.nn.functional.cross_entropy(model(raw), labels).detach()
                self.assertTrue(torch.isfinite(final))
                self.assertLess(float(final), float(initial) * 0.25)


if __name__ == "__main__":
    unittest.main()
