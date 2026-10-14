"""Cluster GPU preflight; run only inside the allocated experiment job."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from eeg_models.models import build_model
from eeg_models.models._shared.covariance import CovarianceCompletion, training_second_moment
from inm.availability import subset_partition
from inm.completion_training import fit_completion, training_schedule


class CovariancePreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Covariance preflight requires the allocated CUDA GPU')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.device = torch.device('cuda')
        cls.metadata = {'modality': 'eeg', 'channels': 22, 'samples': 1000, 'num_classes': 4}

    def completer(self, learned=False):
        moment = torch.eye(22, dtype=torch.float64)
        moment[0, 1] = moment[1, 0] = .8
        module = CovarianceCompletion(learned=learned)
        module.initialize(moment)
        return module.to(self.device)

    def test_conditional_estimate_and_observed_preservation(self):
        module = self.completer()
        x = torch.zeros(2, 22, 4, 8, device=self.device)
        x[:, 0] = 2.
        mask = torch.ones(2, 22, 4, dtype=torch.bool, device=self.device)
        mask[:, 1] = False
        result = module.complete(x, mask)
        expected = 1.6 / ((1 + 1e-6) * (1 + .001))
        torch.testing.assert_close(result[:, 1], torch.full_like(result[:, 1], expected), rtol=1e-6, atol=1e-7)
        observed = mask[..., None].expand_as(x)
        self.assertTrue(torch.equal(result[observed], x[observed]))

    def test_hidden_values_do_not_enter_inference(self):
        module = self.completer()
        x = torch.ones(2, 22, 4, 8, device=self.device)
        mask = torch.ones(2, 22, 4, dtype=torch.bool, device=self.device)
        mask[:, 1, 1:3] = False
        hidden = ~mask[..., None].expand_as(x)
        damaged = x.clone()
        damaged[hidden] = float('nan')
        reference = module.complete(damaged, mask)
        damaged[hidden] = float('inf')
        self.assertTrue(torch.equal(reference, module.complete(damaged, mask)))
        self.assertTrue(bool(torch.isfinite(reference).all()))
        self.assertTrue(torch.equal(reference[~hidden], x[~hidden]))

    def test_full_bypass_and_empty_window_rejection(self):
        module = self.completer()
        x = torch.ones(1, 22, 4, 8, device=self.device)
        mask = torch.ones(1, 22, 4, dtype=torch.bool, device=self.device)
        self.assertIs(module.complete(x, mask), x)
        mask[:, :, 1] = False
        with self.assertRaises(ValueError):
            module.complete(x, mask)

    def test_learned_and_frozen_initialization_and_gradient(self):
        frozen, learned = self.completer(), self.completer(learned=True)
        self.assertFalse(list(frozen.parameters()))
        x = torch.ones(2, 22, 4, 8, device=self.device)
        mask = torch.ones(2, 22, 4, dtype=torch.bool, device=self.device)
        mask[:, 1] = False
        self.assertTrue(torch.equal(frozen.complete(x, mask), learned.complete(x, mask)))
        learned.complete(x, mask)[:, 1].square().mean().backward()
        self.assertIsNotNone(learned.free.grad)
        self.assertTrue(bool(torch.isfinite(learned.free.grad).all()))
        self.assertGreater(float(learned.free.grad.abs().sum()), 0.)
        self.assertTrue(bool((torch.linalg.eigvalsh(learned.covariance()) > 0).all()))

    def test_training_only_moment(self):
        x = torch.ones(3, 22, 8)
        with self.assertRaises(ValueError):
            training_second_moment(x, partition='validation')
        torch.testing.assert_close(training_second_moment(x), torch.ones(22, 22, dtype=torch.float64))

    def test_all_backbone_variants_have_identical_full_input(self):
        x = torch.ones(2, 22, 1000, device=self.device)
        for backbone in ('eegnet', 'signal_transformer'):
            torch.manual_seed(9)
            base = build_model(backbone, self.metadata).to(self.device).eval()
            with torch.no_grad():
                expected = base(x)
            for suffix in ('frozen', 'learned'):
                torch.manual_seed(9)
                variant = build_model(backbone + '_covariance_' + suffix, self.metadata)
                variant.signal_input.completion.initialize(torch.eye(22, dtype=torch.float64))
                variant.to(self.device).eval()
                for key, value in base.state_dict().items():
                    self.assertTrue(torch.equal(value, variant.state_dict()[key]), (backbone, key))
                mask = torch.ones(2, 22, 4, dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    self.assertTrue(torch.equal(expected, variant(x)))
                    self.assertTrue(torch.equal(expected, variant(x, mask)))
                del variant
            del base

    def test_schedule_is_repeatable_and_training_reserved(self):
        cfg = self.config()
        ids = [f'trial-{index}' for index in range(48)]
        mask, order = training_schedule(ids, 3, 0, 1, cfg)
        again, again_order = training_schedule(ids, 3, 0, 1, cfg)
        self.assertTrue(np.array_equal(mask, again))
        self.assertTrue(np.array_equal(order, again_order))
        for trial in mask:
            for column in trial.T:
                self.assertIn(int(column.sum()), (22, 16, 11, 6))
                self.assertIn(subset_partition(column), ('train', 'shared_full'))

    @staticmethod
    def config():
        return {'availability': {'window_samples': 250}, 'data': {'window': 1000},
                'training': {'class_weights': True, 'loss': 'cross_entropy', 'focal_gamma': 2.},
                'covariance': {'learned': {'epochs': 1, 'batch_size': 4, 'learning_rate': .001,
                    'reconstruction_weight': .1, 'anchor_weight': .0001, 'gradient_clip': 1.,
                    'selection': 'fixed_final_epoch', 'training_retained_counts': [22, 16, 11, 6],
                    'training_patterns': ['random_static', 'spatial_static', 'dynamic_random', 'dynamic_spatial']}}}

    def test_supervised_fit_changes_only_covariance(self):
        torch.manual_seed(9)
        model = build_model('eegnet_covariance_learned', self.metadata)
        model.signal_input.completion.initialize(torch.eye(22, dtype=torch.float64))
        model.to(self.device).eval()
        before = copy.deepcopy(model.state_dict())
        x = torch.randn(4, 22, 1000)
        y = np.arange(4, dtype=np.int64)
        cfg = self.config()
        # Force degraded batches in this small gradient-path fixture. The
        # separate schedule test covers the production full/degraded mixture.
        cfg['covariance']['learned']['training_retained_counts'] = [16]
        with tempfile.TemporaryDirectory() as directory:
            record = fit_completion(model, x, y, [f'train-{i}' for i in range(4)],
                lambda mask: torch.where(mask.repeat_interleave(250, dim=2), x, torch.zeros_like(x)),
                cfg, subject=3, seed=0, device=self.device, directory=directory)
            self.assertTrue(record['backbone_unchanged'])
            self.assertTrue(record['full_input_unchanged'])
            self.assertFalse(record['validation_used'])
            self.assertFalse(record['test_used'])
            self.assertEqual(record['selected_epoch'], 1)
            self.assertEqual(json.loads((Path(directory) / 'completion_fit.json').read_text()), record)
            with np.load(Path(directory) / 'completion_training.npz', allow_pickle=False) as data:
                self.assertEqual(data['masks'].shape, (1, 4, 22, 4))
        for key, value in model.state_dict().items():
            if not key.startswith('signal_input.completion.'):
                self.assertTrue(torch.equal(before[key], value), key)
        self.assertFalse(torch.equal(before['signal_input.completion.free'], model.signal_input.completion.free))
        self.assertFalse(model.signal_input.completion.free.requires_grad)


if __name__ == '__main__':
    unittest.main()
