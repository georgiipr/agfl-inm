import unittest

import torch
from torch.nn import functional as F

from inm.encoder_candidates.models import build_model, restore_model
from inm.encoder_candidates.spatial import SpatialEEGNet
from inm.encoder_candidates.transformer import (
    SpatialTransformer,
    sinusoidal_position_encoding,
)


class SpatialTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(101)
        self.raw = torch.randn(2, 22, 4, 250)
        self.full = torch.ones(2, 22, 4, dtype=torch.bool)

    def test_sinusoidal_positions_match_formula_and_fixed_buffer(self):
        actual = sinusoidal_position_encoding(3, 4)
        expected = torch.tensor([[
            [0.0, 1.0, 0.0, 1.0],
            [torch.sin(torch.tensor(1.0)), torch.cos(torch.tensor(1.0)),
             torch.sin(torch.tensor(0.01)), torch.cos(torch.tensor(0.01))],
            [torch.sin(torch.tensor(2.0)), torch.cos(torch.tensor(2.0)),
             torch.sin(torch.tensor(0.02)), torch.cos(torch.tensor(0.02))],
        ]])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "even"):
            sinusoidal_position_encoding(3, 3)
        model = SpatialTransformer(seed=3)
        self.assertIn("position_encoding", dict(model.named_buffers()))
        self.assertFalse(model.position_encoding.requires_grad)
        self.assertEqual(tuple(model.position_encoding.shape), (1, 31, 32))

    def test_reference_front_end_features_and_transformer_shapes(self):
        reference = SpatialEEGNet(seed=23).eval()
        model = SpatialTransformer(seed=23).eval()
        with torch.no_grad():
            expected = reference.forward_features(self.raw)
            actual_front, _ = model._front_end_features(self.raw)
            actual = model.forward_features(self.raw)
            logits = model(self.raw)
        torch.testing.assert_close(actual_front, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual_front.shape), (2, 31, 16))
        self.assertEqual(tuple(actual.shape), (2, 31, 32))
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_seed_reproduces_front_end_independently_of_head_size(self):
        reference = SpatialEEGNet(seed=17)
        first = SpatialTransformer(seed=17)
        second = SpatialTransformer(seed=17)
        front_names = ("temporal_conv", "temporal_bn", "spatial_conv", "spatial_bn",
                       "separable_depthwise", "separable_pointwise", "separable_bn")
        reference_state = reference.encoder.state_dict()
        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())
        for key in first_state:
            torch.testing.assert_close(first_state[key], second_state[key],
                                       rtol=0, atol=0)
        for prefix in front_names:
            ref_items = {key[len(prefix) + 1:]: value for key, value in reference_state.items()
                         if key.startswith(prefix + ".")}
            actual_items = {key[len(prefix) + 1:]: value for key, value in first_state.items()
                            if key.startswith(prefix + ".")}
            self.assertEqual(ref_items.keys(), actual_items.keys())
            for key, value in ref_items.items():
                torch.testing.assert_close(value, actual_items[key], rtol=0, atol=0)
                torch.testing.assert_close(value, second_state[prefix + "." + key],
                                           rtol=0, atol=0)

    def test_finite_gradients_flow_through_front_end_and_attention(self):
        model = SpatialTransformer(seed=2).train()
        loss = F.cross_entropy(model(self.raw), torch.tensor([0, 2]))
        loss.backward()
        for parameter in (model.temporal_conv.weight, model.spatial_conv.weight,
                          model.projection.weight,
                          model.transformer.layers[0].self_attn.in_proj_weight,
                          model.classifier.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_hidden_nan_mask_validation_and_reload(self):
        model = SpatialTransformer(seed=9).eval()
        mask = self.full.clone()
        mask[:, 6, 2] = False
        hidden_nan = self.raw.clone()
        hidden_nan[:, 6, 2] = float("nan")
        expected = model(self.raw, mask)
        actual = model(hidden_nan, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(model(self.raw), model(self.raw, self.full),
                                   rtol=0, atol=0)
        restored = restore_model("spatial_transformer", model.constructor_settings(),
                                 model.state_dict()).eval()
        torch.testing.assert_close(restored(self.raw, mask), expected, rtol=0, atol=0)

        with self.assertRaisesRegex(ValueError, "boolean"):
            model(self.raw, mask.to(torch.int64))
        with self.assertRaisesRegex(ValueError, "boolean"):
            model(self.raw, torch.ones(2, 22, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(self.raw, torch.zeros_like(mask))
        observed_inf = self.raw.clone()
        observed_inf[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(observed_inf)

    def test_factory_round_trips_all_five_arms_and_rejects_provenance_mismatch(self):
        for arm in ("local_control", "local_power", "spatial_eegnet",
                    "spatial_filterbank", "spatial_transformer"):
            with self.subTest(arm=arm):
                original = build_model(arm, seed=5).eval()
                restored = restore_model(arm, original.constructor_settings(),
                                         original.state_dict()).eval()
                if arm == "spatial_transformer":
                    torch.testing.assert_close(restored(self.raw), original(self.raw),
                                               rtol=0, atol=0)
        model = build_model("spatial_transformer", seed=5)
        settings = model.constructor_settings()
        settings["transformer"]["heads"] = 8
        with self.assertRaisesRegex(ValueError, "settings"):
            restore_model("spatial_transformer", settings, model.state_dict())


if __name__ == "__main__":
    unittest.main()
