import unittest

import numpy as np
import torch
from scipy.signal import firwin

from inm.encoder_candidates.models import build_model, restore_model
from inm.encoder_candidates.spatial import SpatialEEGNet, SpatialFilterBank


class SpatialEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.raw = torch.randn(2, 22, 4, 250)
        self.full = torch.ones(2, 22, 4, dtype=torch.bool)

    def test_fixed_fir_coefficients_and_log_variance_match_reference(self):
        model = SpatialFilterBank(seed=4).eval()
        expected_coefficients = np.stack([
            firwin(81, band, pass_zero=False, fs=250, window="hamming")
            for band in model.BANDS_HZ
        ]).astype(np.float32)
        np.testing.assert_allclose(model.fir_coefficients.detach().numpy(),
                                   expected_coefficients, rtol=0, atol=0)
        self.assertFalse(model.fir_coefficients.requires_grad)

        raw = torch.randn(1, 22, 4, 250)
        with torch.no_grad():
            actual = model.forward_features(raw)
            coefficients = model.fir_coefficients[0].reshape(1, 1, -1)
            selected = raw[:, :, :, :].reshape(22 * 4, 1, 250)
            filtered = torch.nn.functional.conv1d(
                torch.nn.functional.pad(selected, (40, 40)), coefficients
            ).reshape(1, 22, 4, 250)
            spatial = torch.einsum("bcwt,oc->bwot", filtered,
                                   model.spatial_filters[0])
            expected = torch.log(spatial.var(-1, unbiased=False) + 1e-6)
        torch.testing.assert_close(actual[:, :, 0], expected, rtol=1e-6, atol=1e-6)

    def test_filter_response_passes_band_and_rejects_stopband_sinusoid(self):
        model = SpatialFilterBank(seed=2).eval()
        coeff = model.fir_coefficients[0]
        time = torch.arange(250, dtype=torch.float32)

        def response_rms(hz):
            signal = torch.sin(2 * torch.pi * hz * time / 250).reshape(1, 1, -1)
            out = torch.nn.functional.conv1d(
                torch.nn.functional.pad(signal, (40, 40)), coeff.reshape(1, 1, -1))
            return out[0, 0, 40:-40].square().mean().sqrt().item()

        passband = response_rms(6)
        stopband = response_rms(40)
        self.assertGreater(passband, 0.5)
        self.assertLess(stopband, passband * 0.02)

    def test_gradient_flow_max_norm_and_fixed_buffer(self):
        model = SpatialFilterBank(seed=7)
        logits = model(self.raw)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        self.assertIsNotNone(model.spatial_filters.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNone(model.fir_coefficients.grad)
        self.assertTrue(torch.isfinite(model.spatial_filters.grad).all())
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

        with torch.no_grad():
            model.spatial_filters.mul_(30)
        model.clip_weights()
        norms = model.spatial_filters.norm(dim=-1)
        self.assertTrue((norms <= 1.0 + 1e-6).all())
        self.assertGreater(float(norms.detach().max()), 0.99)

    def test_masks_select_before_arithmetic_and_logits_reload(self):
        for arm in ("spatial_eegnet", "spatial_filterbank"):
            with self.subTest(arm=arm):
                model = build_model(arm, 19).eval()
                mask = self.full.clone()
                mask[:, 5, 1] = False
                hidden_nan = self.raw.clone()
                hidden_nan[:, 5, 1] = float("nan")
                actual = model(hidden_nan, mask)
                expected = model(self.raw, mask)
                self.assertEqual(tuple(actual.shape), (2, 4))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(model(self.raw), model(self.raw, self.full),
                                           rtol=0, atol=0)
                restored = restore_model(arm, model.constructor_settings(),
                                         model.state_dict()).eval()
                torch.testing.assert_close(restored(self.raw, mask), expected,
                                           rtol=0, atol=0)
                observed_inf = self.raw.clone()
                observed_inf[0, 0, 0, 0] = float("inf")
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    model(observed_inf)
                with self.assertRaisesRegex(ValueError, "at least one"):
                    model(self.raw, torch.zeros_like(mask))

    def test_fir_does_not_cross_window_or_masked_channel_boundaries(self):
        model = SpatialFilterBank(seed=5).eval()
        raw = torch.zeros(1, 22, 4, 250)
        mask = torch.ones(1, 22, 4, dtype=torch.bool)
        first = model.forward_features(raw, mask)
        changed = raw.clone()
        changed[:, :, 2] = torch.randn_like(changed[:, :, 2])
        second = model.forward_features(changed, mask)
        self.assertTrue(torch.equal(first[:, 0], second[:, 0]))
        self.assertTrue(torch.equal(first[:, 1], second[:, 1]))
        self.assertFalse(torch.equal(first[:, 2], second[:, 2]))
        self.assertTrue(torch.equal(first[:, 3], second[:, 3]))

        partial = mask.clone()
        partial[:, 4, 1] = False
        hidden = raw.clone()
        hidden[:, 4, 1] = float("nan")
        torch.testing.assert_close(model(hidden, partial), model(raw, partial),
                                   rtol=0, atol=0)

    def test_spatial_eegnet_wraps_reference_feature_shape_and_constructor(self):
        model = SpatialEEGNet(seed=13).eval()
        self.assertEqual(tuple(model.forward_features(self.raw).shape), (2, 31, 16))
        self.assertEqual(model.constructor_settings()["encoder"]["head_type"], "flatten")
        self.assertEqual(tuple(model(self.raw).shape), (2, 4))
        bad = self.raw.clone()
        bad[0, 1, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(bad)

    def test_factory_rejects_mismatched_spatial_provenance(self):
        model = build_model("spatial_filterbank", 3)
        settings = model.constructor_settings()
        settings["bands_hz"][0] = [3, 8]
        with self.assertRaisesRegex(ValueError, "bands_hz|settings"):
            restore_model("spatial_filterbank", settings, model.state_dict())


if __name__ == "__main__":
    unittest.main()
