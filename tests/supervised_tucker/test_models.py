"""Numerics, differentiation, pairing, masking and calibration isolation."""
import copy
import unittest

import torch
from torch import nn

from inm.tensor_attention import Tucker2
from inm.tensor_temporal.models import TensorTemporal
from inm.supervised_tucker.models import ARM_IDS, build_model, observed_ridge_core, restore_model


class SupervisedTuckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, n=3):
        generator = torch.Generator().manual_seed(43)
        raw = torch.randn(n, 22, 4, 250, generator=generator)
        mask = torch.ones(n, 22, 4, dtype=torch.bool)
        mask[:, 2:9, 1:3] = False
        return raw, mask

    def calibrated(self, arm=ARM_IDS[1], n=3):
        raw, mask = self.inputs(n)
        model = build_model(arm, seed=9).eval()
        report = model.calibrate(raw, tucker_epochs=1)
        return model, raw, mask, report

    def test_solve_matches_explicit_design_and_legacy(self):
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(2, 22, 3, 12, dtype=torch.double, generator=generator)
        u = torch.randn(22, 4, dtype=torch.double, generator=generator)
        v = torch.randn(12, 4, dtype=torch.double, generator=generator)
        mask = torch.rand(2, 22, 3, generator=generator) > 0.4
        mask[:, 0] = True
        actual = observed_ridge_core(x, mask, u, v)
        for i in range(2):
            for t in range(3):
                seen = mask[i, :, t]
                design = torch.einsum("cr,fs->cfrs", u[seen], v).reshape(-1, 16)
                penalty = int(seen.sum()) * 12 * .001
                expected = torch.linalg.solve(design.T @ design + penalty * torch.eye(16, dtype=torch.double),
                                              design.T @ x[i, seen, t].flatten()).reshape(4, 4)
                torch.testing.assert_close(actual[i, :, t], expected, rtol=1e-10, atol=1e-10)
        legacy = Tucker2(22, 12).double()
        legacy.U.copy_(u)
        legacy.V.copy_(v)
        legacy.fitted.fill_(True)
        torch.testing.assert_close(actual, legacy.encode(x, mask), rtol=1e-10, atol=1e-10)

    def test_partially_observed_solve_passes_gradcheck(self):
        generator = torch.Generator().manual_seed(13)
        x = torch.randn(1, 3, 2, 3, dtype=torch.double, generator=generator, requires_grad=True)
        u = torch.randn(3, 2, dtype=torch.double, generator=generator, requires_grad=True)
        v = torch.randn(3, 2, dtype=torch.double, generator=generator, requires_grad=True)
        mask = torch.tensor([[[True, False], [False, True], [True, True]]])
        self.assertTrue(torch.autograd.gradcheck(lambda a, b, c: observed_ridge_core(a, mask, b, c),
                                                (x, u, v), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_rank_deficient_factors_and_single_electrode_remain_differentiable(self):
        x = torch.randn(1, 3, 2, 3, dtype=torch.double, requires_grad=True)
        u = torch.ones(3, 2, dtype=torch.double, requires_grad=True)
        v = torch.ones(3, 2, dtype=torch.double, requires_grad=True)
        mask = torch.zeros(1, 3, 2, dtype=torch.bool)
        mask[:, 0] = True
        core = observed_ridge_core(x, mask, u, v)
        core.square().sum().backward()
        for value in (core, x.grad, u.grad, v.grad):
            self.assertTrue(torch.isfinite(value).all())

    def test_both_arms_start_identical_to_legacy_calibration_and_head(self):
        raw, mask = self.inputs()
        models = [build_model(arm, 9).eval() for arm in ARM_IDS]
        legacy = TensorTemporal("fixed_spectral_tucker", 9).eval()
        legacy_report = legacy.calibrate(raw, tucker_epochs=1)
        for model in models:
            before = model.classifier.weight.clone()
            report = model.calibrate(raw, tucker_epochs=1)
            torch.testing.assert_close(before, model.classifier.weight, rtol=0, atol=0)
            torch.testing.assert_close(model.U, legacy.tucker.U, rtol=0, atol=0)
            torch.testing.assert_close(model.V, legacy.tucker.V, rtol=0, atol=0)
            self.assertEqual(report["factor_history"], legacy_report["factor_history"])
            torch.testing.assert_close(model(raw, mask), legacy(raw, mask), rtol=2e-6, atol=2e-6)
        self.assertEqual(models[0].state_dict().keys(), models[1].state_dict().keys())
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(models[0](raw, mask), models[1](raw, mask), rtol=0, atol=0)

    def test_hidden_nan_forward_and_gradients_match_zero_placeholders(self):
        for arm in ARM_IDS:
            with self.subTest(arm=arm):
                model, raw, mask, _ = self.calibrated(arm)
                hidden = torch.where(mask[..., None], raw, torch.full_like(raw, float("nan"))).requires_grad_()
                zero = torch.where(mask[..., None], raw, torch.zeros_like(raw)).requires_grad_()
                logits = model(hidden, mask)
                logits.square().sum().backward()
                parameter_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                other = model(zero, mask)
                other.square().sum().backward()
                torch.testing.assert_close(logits, other, rtol=0, atol=0)
                torch.testing.assert_close(hidden.grad, zero.grad, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertEqual(float(hidden.grad[~mask].abs().sum()), 0)
                for name, p in model.named_parameters():
                    self.assertTrue(torch.isfinite(p.grad).all())
                    torch.testing.assert_close(parameter_grads[name], p.grad, rtol=0, atol=0)

    def test_frozen_buffers_and_supervised_gradient_reachability(self):
        for arm in ARM_IDS:
            model, raw, mask, _ = self.calibrated(arm)
            nn.functional.cross_entropy(model(raw, mask), torch.arange(3)).backward()
            names = dict(model.named_parameters())
            if arm == ARM_IDS[0]:
                self.assertNotIn("U", names)
                self.assertNotIn("V", names)
                self.assertIn("U", dict(model.named_buffers()))
                self.assertIn("V", dict(model.named_buffers()))
            else:
                for name in ("U", "V"):
                    self.assertTrue(torch.isfinite(names[name].grad).all())
                    self.assertGreater(float(names[name].grad.abs().sum()), 0)
            self.assertGreater(float(names["classifier.weight"].grad.abs().sum()), 0)

    def test_optimization_and_post_update_checkpoint(self):
        model, raw, mask, _ = self.calibrated(n=8)
        labels = torch.arange(8) % 4
        before_u, before_v = model.U.detach().clone(), model.V.detach().clone()
        initial = nn.functional.cross_entropy(model(raw, mask), labels).detach()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=0)
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(raw, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            model.clip_weights()
        final = nn.functional.cross_entropy(model(raw, mask), labels).detach()
        self.assertLess(float(final), float(initial) * .4)
        self.assertFalse(torch.equal(before_u, model.U))
        self.assertFalse(torch.equal(before_v, model.V))
        for factor in (model.U, model.V):
            torch.testing.assert_close(factor.norm(dim=0), torch.ones(4), rtol=1e-6, atol=1e-6)
        restored = restore_model(ARM_IDS[1], model.constructor_settings(), model.state_dict()).eval()
        torch.testing.assert_close(model(raw, mask), restored(raw, mask), rtol=0, atol=0)
        wrong = copy.deepcopy(model.constructor_settings())
        wrong["tensor"]["tucker_ridge"] = .1
        with self.assertRaises(ValueError):
            restore_model(ARM_IDS[1], wrong, model.state_dict())

    def test_calibration_statistics_immutable_and_rng_preserved(self):
        raw, mask = self.inputs()
        torch.manual_seed(221)
        state = torch.random.get_rng_state().clone()
        for arm in ARM_IDS:
            model = build_model(arm, 9).eval()
            report = model.calibrate(raw, tucker_epochs=1)
            torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
            self.assertEqual(report["training_trials"], 3)
            self.assertEqual(report["factor_epochs"], 1)
            spectral = model._spectral(raw)
            torch.testing.assert_close(model.spectral_mean, spectral.mean((0, 2), keepdim=True))
            torch.testing.assert_close(model.spectral_std, spectral.std((0, 2), unbiased=False, keepdim=True).clamp_min(1e-6))
            before = {k: v.clone() for k, v in model.state_dict().items()}
            model(raw * 20, mask)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                model.calibrate(raw * 20)
            with self.assertRaises(TypeError):
                model.calibrate(raw, validation_raw=raw)

    def test_optimizer_and_clipping_leave_frozen_factors_unchanged(self):
        model, raw, _, _ = self.calibrated(ARM_IDS[0])
        u, v = model.U.clone(), model.V.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model(raw).square().sum().backward()
        optimizer.step()
        model.clip_weights()
        torch.testing.assert_close(u, model.U, rtol=0, atol=0)
        torch.testing.assert_close(v, model.V, rtol=0, atol=0)

    def test_invalid_masks_observed_nans_and_uncalibrated_models_rejected(self):
        raw, mask = self.inputs()
        model = build_model(ARM_IDS[1])
        with self.assertRaises(RuntimeError):
            model(raw)
        model.calibrate(raw, tucker_epochs=1)
        empty = mask.clone()
        empty[:, :, 0] = False
        for x, m in ((raw, empty), (raw, mask.float()), (raw.double(), mask)):
            with self.assertRaises(ValueError):
                model(x, m)
        raw[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(raw, mask)


if __name__ == "__main__":
    unittest.main()
