import tempfile
import unittest
from pathlib import Path

import torch

from inm.baselines.eegnet import EEGNetClassifier
from inm.baselines.protocol import arms, load_config, tasks
from inm.baselines.training import _atomic_torch_save, checkpoint_payload, load_classifier


ROOT = Path(__file__).resolve().parents[2]


class TemporalHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.model = EEGNetClassifier(mask_conditioned=True, head_type="temporal_conv",
            temporal_head_width=8, temporal_head_kernel=3, temporal_head_dilation=1,
            dropout=0.0).cpu()
        self.raw = torch.randn(2, 22, 4, 250)
        self.mask = torch.ones(2, 22, 4, dtype=torch.bool)

    def test_temporal_logits_gradients_and_parameter_count(self):
        self.model.train()
        logits = self.model(self.raw, self.mask)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(self.model.temporal_head.kernel_size), (3,))
        self.assertEqual(self.model.temporal_head.in_channels, self.model.f2)
        self.assertEqual(self.model.temporal_head.out_channels, 8)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 2236)
        logits.square().mean().backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(self.model.temporal_head.weight.grad.abs().sum()), 0)

    def test_hidden_values_and_gradients_are_invariant(self):
        self.model.eval()
        mask = self.mask.clone()
        mask[:, 1::2, 1] = False
        baseline = self.model(self.raw, mask)
        changed = self.raw.clone()
        changed[~mask] = float("nan")
        torch.testing.assert_close(self.model(changed, mask), baseline)
        changed[~mask] = 1e6
        torch.testing.assert_close(self.model(changed, mask), baseline)

        raw = self.raw.clone().requires_grad_(True)
        self.model(raw, mask).sum().backward()
        self.assertTrue(torch.equal(raw.grad[~mask], torch.zeros_like(raw.grad[~mask])))
        self.assertTrue(torch.isfinite(raw.grad[mask]).all())

    def test_checkpoint_roundtrip_preserves_temporal_logits(self):
        self.model.eval()
        expected = self.model(self.raw, self.mask)
        metadata = {"selected_epoch": 1, "head_type": "temporal_conv"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            _atomic_torch_save(checkpoint_payload(self.model, metadata), path)
            restored, payload = load_classifier(path, device="cpu")
        self.assertEqual(payload["constructor"]["head_type"], "temporal_conv")
        self.assertEqual(payload["metadata"]["head_type"], "temporal_conv")
        torch.testing.assert_close(restored.eval()(self.raw, self.mask), expected)

    def test_ablation_config_enumerates_disjoint_head_arms(self):
        cfg = load_config(ROOT / "configs/baselines-temporal.json")
        matrix = arms(cfg)
        self.assertEqual(len(tasks(cfg)), 27)
        self.assertEqual(len(matrix), 4)
        self.assertEqual([arm["head_type"] for arm in matrix], [
            "flatten", "temporal_conv", "flatten", "temporal_conv"])
        self.assertEqual(len({arm["name"] for arm in matrix}), 4)
        self.assertEqual(len({arm["name"] for arm in matrix if arm["head_type"] == "flatten"}), 2)
        self.assertEqual(len({arm["name"] for arm in matrix if arm["head_type"] == "temporal_conv"}), 2)
        for regime in ("full", "mixed"):
            pair = [arm for arm in matrix if arm["regime"] == regime]
            self.assertEqual({arm["head_type"] for arm in pair}, {"flatten", "temporal_conv"})
            self.assertEqual(pair[0]["coverage"], pair[1]["coverage"])

    def test_flatten_default_and_legacy_config_matrices_are_unchanged(self):
        default = load_config(ROOT / "configs/baselines.json")
        self.assertEqual([arm["name"] for arm in arms(default)], [
            "eegnet_reference__full", "masked_eegnet__full",
            "masked_eegnet__mixed", "covariance__full"])
        model = EEGNetClassifier(mask_conditioned=True)
        self.assertEqual(model.head_type, "flatten")
        self.assertIsNone(model.temporal_head)


if __name__ == "__main__":
    unittest.main()
