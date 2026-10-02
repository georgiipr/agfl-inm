import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from inm.tensor_followup import feature_digest, load_config, matrix, paired_identity
from inm.tensor_attention import Tucker2
from inm.training import fit


ROOT = Path(__file__).resolve().parents[2]


class TensorFollowupTests(unittest.TestCase):
    def test_declared_matrix_and_output_are_separate(self):
        cfg = load_config(ROOT / 'configs' / 'tensor-followup.json')
        self.assertEqual(len(matrix(cfg)), 6)
        self.assertEqual({row['attention'] for row in matrix(cfg)}, {'mha'})
        self.assertEqual({row['representation'] for row in matrix(cfg)},
                         {'baseline', 'tensor_completion', 'tensor_core'})
        self.assertNotEqual(cfg['output_dir'], str((ROOT / 'results' / 'inm-v2').resolve()))
        self.assertEqual(set(cfg['tensor_followup']['selection']['patterns']), {
            'full_22', 'random_static_16', 'dynamic_random_16',
            'random_static_6', 'dynamic_random_6'})
        cfg['mha_controls'] = cfg['tensor_followup']['available_controls']
        self.assertEqual(len(matrix(cfg)), 10)

    def test_factor_freeze_mask_isolation_completion_and_pairing(self):
        torch.manual_seed(7)
        x = torch.randn(5, 4, 2, 3)
        full = torch.ones(5, 4, 2, dtype=torch.bool)
        tensor = Tucker2(4, 3, rank_channels=2, rank_features=2,
                         ridge=1e-3, fit_epochs=1, fit_batch_size=5)
        tensor.fit(x, full)
        u, v = tensor.U.clone(), tensor.V.clone()

        mask = full[:2].clone()
        mask[:, 1, 0] = False
        hidden_nan = x[:2].clone()
        hidden_nan[:, 1, 0] = float('nan')
        safe = x[:2].clone()
        safe[:, 1, 0] = 12345
        core_nan = tensor.encode(hidden_nan, mask)
        core_value = tensor.encode(safe, mask)
        self.assertTrue(torch.allclose(core_nan, core_value))
        self.assertFalse(core_nan.requires_grad)
        completed = tensor.complete(hidden_nan, mask)
        self.assertTrue(torch.equal(completed[mask], hidden_nan[mask]))

        left = paired_identity(feature_digest(x), 'calibration-hash', 'mask-hash')
        right = paired_identity(feature_digest(x.clone()), 'calibration-hash', 'mask-hash')
        self.assertEqual(left, right)
        self.assertTrue(torch.equal(tensor.U, u) and torch.equal(tensor.V, v))
        self.assertFalse(tensor.U.requires_grad or tensor.V.requires_grad)

        class TensorHead(torch.nn.Module):
            def __init__(self, fixed_tensor):
                super().__init__()
                self.tensor = fixed_tensor
                self.head = torch.nn.Linear(2, 2)

            def forward(self, features, available):
                latent = self.tensor.encode(features, available)
                return self.head(latent.mean(dim=(1, 2)))

        classifier = TensorHead(tensor)
        optimizer = torch.optim.SGD((p for p in classifier.parameters() if p.requires_grad), lr=0.1)
        optimizer.zero_grad()
        classifier(x[:2], full[:2]).sum().backward()
        optimizer.step()
        self.assertTrue(torch.equal(tensor.U, u) and torch.equal(tensor.V, v))

    def test_fit_selects_from_named_heldout_validation_banks(self):
        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 4)

            def forward(self, values, mask):
                return self.linear(values)

        torch.manual_seed(11)
        x = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        train_masks = torch.ones(8, 22, 4, dtype=torch.bool)
        heldout_masks = torch.ones(4, 22, 4, dtype=torch.bool)
        options = {'class_weights': False, 'loss': 'cross_entropy', 'focal_gamma': 2.0,
                   'learning_rate': 0.01, 'weight_decay': 0.0, 'epochs': 2,
                   'warmup_epochs': 0, 'batch_size': 4, 'gradient_clip': 1.0,
                   'minimum_epochs': 1, 'patience': 0}
        with TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / 'history.json'
            selection = fit(TinyClassifier(), x, y, x[:4], y[:4],
                            lambda epoch: train_masks, heldout_masks, options,
                            seed=3, device='cpu', history_path=history_path,
                            description='tiny', validation_masks=[
                                ('full_22', heldout_masks),
                                ('random_static_16', heldout_masks),
                                ('random_static_16', heldout_masks),
                            ])
            history = json.loads(history_path.read_text())
        self.assertEqual(selection['selection'], 'mean_validation_balanced_accuracy_then_mean_log_loss')
        selected = history[selection['best_epoch'] - 1]['validation_banks']
        self.assertEqual(set(selected), {'full_22', 'random_static_16'})


if __name__ == '__main__':
    unittest.main()
