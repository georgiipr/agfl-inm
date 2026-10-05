import unittest

import torch

from inm.encoder_candidates.local import LocalClassifier, LocalPowerEncoder
from inm.encoder_candidates.models import build_model, restore_model


class LocalEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.raw = torch.randn(2, 22, 4, 250)
        self.mask = torch.ones(2, 22, 4, dtype=torch.bool)

    def test_log_variance_matches_direct_population_reference_and_amplitude_scaling(self):
        torch.manual_seed(2)
        encoder = LocalPowerEncoder(channels=1, windows=1, kernels=(7,),
                                    filters_per_branch=1, layout="B1W1F1")
        signal = torch.randn(1, 1, 1, 250)
        with torch.no_grad():
            actual = encoder.forward_features(signal)
            response = torch.nn.functional.conv1d(
                torch.nn.functional.pad(signal.reshape(1, 1, 250), (3, 3)),
                encoder.branches[0].weight)
            expected = (response.var(-1, unbiased=False) + 1e-6).log().reshape(1, 1, 1, 1)
            scaled = encoder.forward_features(signal * 3.0)
            scaled_response = torch.nn.functional.conv1d(
                torch.nn.functional.pad((signal * 3).reshape(1, 1, 250), (3, 3)),
                encoder.branches[0].weight)
            expected_scaled = (scaled_response.var(-1, unbiased=False) + 1e-6).log()
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(scaled.flatten(), expected_scaled.flatten(), rtol=0, atol=1e-7)
        self.assertGreater((scaled - actual).item(), 1.0)

    def test_constant_inputs_use_epsilon_and_have_finite_gradients(self):
        encoder = LocalPowerEncoder(channels=1, windows=1, kernels=(1,),
                                    filters_per_branch=1, layout="B1W1F1")
        with torch.no_grad():
            encoder.branches[0].weight.fill_(2.0)
        raw = torch.full((2, 1, 1, 250), 4.0, requires_grad=True)
        output = encoder.forward_features(raw)
        torch.testing.assert_close(output, torch.full_like(output, torch.log(torch.tensor(1e-6))),
                                   rtol=0, atol=1e-6)
        output.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in encoder.parameters()))

        model = LocalClassifier("local_power", seed=17)
        logits = model(torch.randn(2, 22, 4, 250))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_local_features_do_not_mix_channels_or_windows(self):
        for kind in ("local_control", "local_power"):
            with self.subTest(kind=kind):
                model = LocalClassifier(kind, seed=11).eval()
                base = model.forward_features(self.raw)
                changed = self.raw.clone()
                changed[0, 3, 2] += 5.0
                perturbed = model.forward_features(changed)
                delta = (perturbed - base).abs().sum(dim=-1)
                affected = delta > 0
                self.assertTrue(affected[0, 3, 2])
                self.assertEqual(int(affected.sum()), 1)
                self.assertFalse(torch.equal(perturbed[0, 3, 2], base[0, 3, 2]))

    def test_masks_select_before_arithmetic_and_validate_inputs(self):
        for kind in ("local_control", "local_power"):
            with self.subTest(kind=kind):
                model = LocalClassifier(kind, seed=3).eval()
                mask = self.mask.clone(); mask[:, 4, 1] = False
                hidden_nan = self.raw.clone(); hidden_nan[:, 4, 1] = float("nan")
                logits = model(hidden_nan, mask)
                self.assertTrue(torch.isfinite(logits).all())
                torch.testing.assert_close(model(self.raw, self.mask), model(self.raw), rtol=0, atol=0)
                self.assertEqual(model.forward_features(hidden_nan, mask)[:, 4, 1].count_nonzero().item(), 0)

                bad = mask.clone(); bad[:, :, 0] = False
                with self.assertRaisesRegex(ValueError, "at least one"):
                    model(self.raw, bad)
                with self.assertRaisesRegex(ValueError, "boolean"):
                    model(self.raw, mask.float())
                observed_nan = self.raw.clone(); observed_nan[0, 0, 0, 0] = float("nan")
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    model(observed_nan)
                with self.assertRaisesRegex(ValueError, "shape"):
                    model(torch.zeros(2, 22, 1000))

    def test_batch_size_invariance(self):
        model = LocalClassifier("local_power", seed=19).eval()
        together = model(self.raw)
        separate = torch.cat([model(self.raw[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(together, separate, rtol=1e-6, atol=1e-6)

    def test_control_and_power_heads_have_identical_initialization(self):
        torch.manual_seed(901)
        rng_before = torch.random.get_rng_state().clone()
        control = LocalClassifier("local_control", seed=23)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        power = LocalClassifier("local_power", seed=23)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        self.assertEqual(control.head.state_dict().keys(), power.head.state_dict().keys())
        for name, value in control.head.state_dict().items():
            torch.testing.assert_close(value, power.head.state_dict()[name], rtol=0, atol=0)

    def test_control_feature_shape_and_feature_freeze_survives_parent_train(self):
        model = LocalClassifier("local_control", seed=5)
        self.assertEqual(tuple(model.forward_features(self.raw).shape), (2, 22, 4, 32))
        model.freeze_features().train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        model.unfreeze_features().train()
        self.assertTrue(model.encoder.training)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))

    def test_state_reload_and_factory_constructor_metadata(self):
        model = build_model("local_power", 29).eval()
        original = model(self.raw)
        settings = model.constructor_settings()
        restored = restore_model("local_power", settings, model.state_dict()).eval()
        torch.testing.assert_close(restored(self.raw), original, rtol=0, atol=0)
        self.assertEqual(restored.constructor_settings(), settings)
        with self.assertRaisesRegex(ValueError, "does not match"):
            restore_model("local_control", settings, model.state_dict())
        transformer = build_model("spatial_transformer", 0)
        self.assertEqual(transformer.constructor_settings()["seed"], 0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_model("invented", 0)


if __name__ == "__main__":
    unittest.main()
