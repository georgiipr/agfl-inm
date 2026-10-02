import unittest

import torch

from inm.baselines.eegnet import EEGNetClassifier


class EEGNetClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = EEGNetClassifier(dropout=0.25).cpu()
        self.raw = torch.randn(2, 22, 4, 250)
        self.mask = torch.ones(2, 22, 4, dtype=torch.bool)

    def test_forward_backward_and_spatial_kernel(self):
        logits = self.model(self.raw, self.mask)
        self.assertEqual(tuple(logits.shape), (2, 4))
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))
        spatial_grad = self.model.spatial_conv.weight.grad
        self.assertIsNotNone(spatial_grad)
        self.assertGreater(float(spatial_grad.abs().sum()), 0.0)
        self.assertEqual(tuple(self.model.spatial_conv.kernel_size), (22, 1))
        self.assertEqual(self.model.spatial_conv.groups, self.model.f1)

    def test_all_observed_matches_omitted_mask(self):
        self.model.eval()
        torch.testing.assert_close(self.model(self.raw), self.model(self.raw, self.mask))

    def test_hidden_nan_and_random_values_do_not_change_eval_logits(self):
        self.model.eval()
        mask = self.mask.clone()
        mask[:, 1::2, 1] = False
        changed = self.raw.clone()
        changed[~mask] = float("nan")
        baseline = self.model(self.raw, mask)
        torch.testing.assert_close(self.model(changed, mask), baseline)

        changed[~mask] = torch.randn_like(changed[~mask]) * 1e6
        torch.testing.assert_close(self.model(changed, mask), baseline)

    def test_hidden_samples_have_zero_input_gradient(self):
        self.model.eval()
        mask = self.mask.clone()
        mask[:, 0, :] = False
        raw = self.raw.clone().requires_grad_(True)
        self.model(raw, mask).sum().backward()
        self.assertTrue(torch.equal(raw.grad[~mask], torch.zeros_like(raw.grad[~mask])))
        self.assertTrue(torch.isfinite(raw.grad[mask]).all())

    def test_conditioned_model_appends_original_flags(self):
        model = EEGNetClassifier(mask_conditioned=True, dropout=0.0)
        self.assertEqual(model.classifier.in_features,
                         model.feature_size + 22 * 4)
        self.assertEqual(model.forward_features(self.raw, self.mask).shape,
                         (2, model.feature_steps, model.f2))
        self.assertEqual(model.constructor_settings()["mask_conditioned"], True)

    def test_state_dict_roundtrip_preserves_logits(self):
        self.model.eval()
        logits = self.model(self.raw, self.mask)
        restored = EEGNetClassifier(**self.model.constructor_settings()).cpu().eval()
        restored.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(restored(self.raw, self.mask), logits)

    def test_invalid_masks_and_observed_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(self.raw, torch.ones(2, 22, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.model(self.raw, torch.ones(2, 22, 4, dtype=torch.float32))
        empty = self.mask.clone()
        empty[0, :, 2] = False
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.model(self.raw, empty)
        nonfinite = self.raw.clone()
        nonfinite[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.model(nonfinite, self.mask)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(torch.randn(2, 22, 1000), self.mask)
        with self.assertRaisesRegex(ValueError, "floating point"):
            self.model(torch.ones(2, 22, 4, 250, dtype=torch.int64), self.mask)
        with self.assertRaisesRegex(ValueError, "same dtype"):
            self.model(self.raw.double(), self.mask)
        wrong_device = torch.ones(2, 22, 4, dtype=torch.bool, device="meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            self.model(self.raw, wrong_device)


if __name__ == "__main__":
    unittest.main()
