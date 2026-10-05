import unittest

import torch
from torch import nn

from inm.completion_transformer import CompletionAdapter, CovarianceCompleter
from inm.encoder_candidates.spatial import SpatialEEGNet
from inm.encoder_candidates.transformer import SpatialTransformer


class _MustNotComplete(nn.Module):
    def complete(self, x, mask):
        raise AssertionError("full-input path must bypass completion")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(22)
        self.raw = torch.randn(2, 22, 4, 250, generator=generator)
        self.mask = torch.ones(2, 22, 4, dtype=torch.bool)
        self.mask[:, 3, :] = False

    def test_zero_adapter_logits_equal_original_and_preserves_all_state(self):
        for model in (SpatialEEGNet(seed=3), SpatialTransformer(seed=3)):
            model.eval()
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            adapter = CompletionAdapter(model, "zero")
            self.assertFalse(adapter.training)
            torch.testing.assert_close(adapter(self.raw, self.mask), model(self.raw, self.mask), rtol=0, atol=0)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)

    def test_original_availability_flags_reach_both_classifiers(self):
        for model in (SpatialEEGNet(seed=13), SpatialTransformer(seed=13)):
            model.eval()
            classifier = model.encoder.classifier if isinstance(model, SpatialEEGNet) else model.classifier
            captured = []
            handle = classifier.register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach().clone()))
            adapter = CompletionAdapter(model, "zero")
            adapter(self.raw, self.mask)
            handle.remove()
            self.assertEqual(len(captured), 1)
            torch.testing.assert_close(captured[0][:, -88:], self.mask.reshape(2, -1).float(), rtol=0, atol=0)

    def test_full_input_equivalence_and_completion_bypass_for_every_strategy(self):
        full_mask = torch.ones_like(self.mask)
        for model in (SpatialEEGNet(seed=19), SpatialTransformer(seed=19)):
            model.eval()
            expected = model(self.raw, full_mask)
            zero = CompletionAdapter(model, "zero")
            covariance = CompletionAdapter(model, "covariance", _MustNotComplete())
            tucker = CompletionAdapter(model, "tucker", _MustNotComplete())
            for adapter in (zero, covariance, tucker):
                torch.testing.assert_close(adapter(self.raw, full_mask), expected,
                                           rtol=1e-6, atol=1e-7)

    def test_hidden_nan_logits_and_covariance_completion_influence(self):
        generator = torch.Generator().manual_seed(5)
        training = torch.randn(6, 22, 4, 250, generator=generator)
        covariance = CovarianceCompleter().fit(training)
        for model in (SpatialEEGNet(seed=9), SpatialTransformer(seed=9)):
            model.eval()
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            zero_adapter = CompletionAdapter(model, "zero")
            completed_adapter = CompletionAdapter(model, "covariance", covariance)
            hidden_nan = self.raw.clone()
            hidden_nan[~self.mask] = float("nan")
            hidden_other = self.raw.clone()
            hidden_other[~self.mask] = 123.0
            zero_logits = zero_adapter(hidden_nan, self.mask)
            completed_logits = completed_adapter(hidden_nan, self.mask)
            torch.testing.assert_close(zero_logits, zero_adapter(hidden_other, self.mask), rtol=0, atol=0)
            torch.testing.assert_close(completed_logits, completed_adapter(hidden_other, self.mask), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(completed_logits).all())
            self.assertFalse(torch.equal(completed_logits, zero_logits))
            self.assertFalse(completed_logits.requires_grad)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)

    def test_caller_cannot_reenable_training_mode_on_backbone(self):
        model = SpatialTransformer(seed=29)
        adapter = CompletionAdapter(model, "zero")
        self.assertFalse(adapter.training)
        with self.assertRaisesRegex(RuntimeError, "replay-only"):
            adapter.train()
        model.train()
        with self.assertRaisesRegex(RuntimeError, "must remain in eval mode"):
            adapter(self.raw, self.mask)
        adapter.eval()
        self.assertFalse(model.training)
        self.assertTrue(torch.isfinite(adapter(self.raw, self.mask)).all())


if __name__ == "__main__":
    unittest.main()
