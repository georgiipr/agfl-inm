"""Public process boundaries and independently replayed synthetic evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inm.task_driven_completion.audit import audit_task, prediction_metrics, task_contrasts
from inm.task_driven_completion.protocol import CONDITIONS, STRATEGY_IDS, file_sha256, load_config, study_identity
from inm.task_driven_completion.reporting import aggregate
from inm.task_driven_completion.study import _atomic_json, _atomic_npz, _verify_complete_task

ROOT = Path(__file__).resolve().parents[2]


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        declaration = json.loads((ROOT / 'configs/task-driven-completion.json').read_text())
        declaration.update(synthetic=True, subjects=[1], seeds=[0], data_dir='data',
                           candidate_source_dir='candidate', split_source_dir='split', output_dir='output')
        cls.config = cls.root / 'config.json'
        cls.config.write_text(json.dumps(declaration))
        cls.output = cls.root / 'output'
        cls.smoke_args = ['--config', str(cls.config), '--smoke', '--output-dir', str(cls.output)]
        result = cls.process('inm.task_driven_completion', cls.smoke_args)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.fresh = json.loads(result.stdout)
        cls.cfg = load_config(cls.config, allow_existing_output=True)
        cls.path = cls.output / 'tasks/A01_seed_0'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def process(module, args):
        return subprocess.run([sys.executable, '-m', module, *args], cwd=ROOT,
                              env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                              capture_output=True, text=True, timeout=90)

    def hashes(self):
        return {str(p.relative_to(self.path)): file_sha256(p)
                for p in self.path.rglob('*') if p.is_file()}

    def test_public_fresh_resume_audit_in_distinct_processes(self):
        self.assertEqual(self.fresh['cells'], 105)
        self.assertEqual(set(self.fresh['strategies']), set(STRATEGY_IDS))
        self.assertEqual(self.fresh['supervised_completion_fits'], 2)
        before = self.hashes()
        resumed = self.process('inm.task_driven_completion', self.smoke_args + ['--resume-smoke'])
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(json.loads(resumed.stdout), self.fresh)
        audited = self.process('inm.task_driven_completion.audit',
            ['--config', str(self.config), '--synthetic-smoke', '--output-dir', str(self.output)])
        self.assertEqual(audited.returncode, 0, audited.stderr)
        report = json.loads(audited.stdout)
        self.assertTrue(report['synthetic'])
        self.assertEqual(report['replayed_cells'], 105)
        self.assertTrue(report['backbone_matches_source'])
        self.assertEqual(len(report['task_contrasts']), 5)
        self.assertIsNone(report['cohort_estimates'])
        self.assertEqual(before, self.hashes())

    def test_public_changed_config_identity_rejected_without_mutation(self):
        original = self.config.read_bytes()
        before = self.hashes()
        try:
            changed = json.loads(original)
            changed['name'] += '-changed-identity'
            self.config.write_text(json.dumps(changed))
            result = self.process('inm.task_driven_completion', self.smoke_args + ['--resume-smoke'])
            self.assertEqual(result.returncode, 2)
            self.assertIn('Changed source/config/packages/history identity', result.stderr)
            result = self.process('inm.task_driven_completion.audit',
                ['--config', str(self.config), '--synthetic-smoke', '--output-dir', str(self.output)])
            self.assertEqual(result.returncode, 2)
            self.assertIn('identity differs', result.stderr)
            self.assertEqual(before, self.hashes())
        finally:
            self.config.write_bytes(original)

    def test_rehashed_selected_state_tampering_fails_actual_replay(self):
        path = self.path / 'covariance_learned/selected.npz'
        manifest = self.path / 'task.json'
        original, original_manifest = path.read_bytes(), manifest.read_bytes()
        try:
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
            arrays['free'][1] += .5
            _atomic_npz(path, **arrays)
            task = json.loads(original_manifest)
            task['artifacts']['covariance_learned/selected.npz'] = file_sha256(path)
            self.assertTrue(task['exact_selected_state_replay'])
            _atomic_json(manifest, task)
            # This is well-formed, rehashed state; schema verification alone passes.
            _verify_complete_task(self.path, study_identity(self.cfg), synthetic=True)
            with self.assertRaisesRegex(ValueError, 'selected-state predictions differ'):
                audit_task(self.cfg, 0)
        finally:
            path.write_bytes(original)
            manifest.write_bytes(original_manifest)

    def test_rehashed_backbone_tampering_rejected_against_source(self):
        path, manifest = self.path / 'backbone.npz', self.path / 'task.json'
        original, original_manifest = path.read_bytes(), manifest.read_bytes()
        try:
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
            key = next(k for k in arrays if k.endswith('weight') and arrays[k].dtype.kind == 'f')
            arrays[key].flat[0] += .25
            _atomic_npz(path, **arrays)
            task = json.loads(original_manifest)
            task['artifacts']['backbone.npz'] = file_sha256(path)
            _atomic_json(manifest, task)
            _verify_complete_task(self.path, study_identity(self.cfg), synthetic=True)
            with self.assertRaisesRegex(ValueError, 'backbone snapshot differs'):
                audit_task(self.cfg, 0)
        finally:
            path.write_bytes(original)
            manifest.write_bytes(original_manifest)

    def test_audit_never_calls_calibration_or_fitting(self):
        before = self.hashes()
        with patch('inm.task_driven_completion.completion.fit_initialization', side_effect=AssertionError('fit')), \
             patch('inm.task_driven_completion.training.fit_completion', side_effect=AssertionError('fit')), \
             patch('inm.task_driven_completion.training.evaluate', side_effect=AssertionError('shared evaluation')):
            report = audit_task(self.cfg, 0)
        self.assertEqual(report['new_fits'], 0)
        self.assertEqual(report['replayed_cells'], 105)
        self.assertEqual(before, self.hashes())

    def test_independent_metrics_use_equal_class_weight_with_unequal_counts(self):
        labels = np.array([0] * 10 + [1, 2, 3])
        probabilities = np.full((13, 4), .1)
        probabilities[:, 0] = .7
        metrics = prediction_metrics(labels, probabilities)
        self.assertAlmostEqual(metrics['balanced_accuracy'], .25)
        self.assertAlmostEqual(metrics['log_loss'], -(10 * np.log(.7) + 3 * np.log(.1)) / 13)

    def test_unequal_trial_counts_negative_effects_and_incomplete_cohort(self):
        tasks = []
        for subject, count, effects in ((1, 400, (.20, .10)), (2, 4, (-.30, -.20))):
            for seed, effect in enumerate(effects):
                cells = []
                for strategy in STRATEGY_IDS:
                    for condition in CONDITIONS:
                        for repeat in range(condition['repeats']):
                            offset = .01 * repeat if condition['name'] != 'full_22' else 0
                            cells.append(dict(strategy=strategy, condition=condition['name'], repeat=repeat,
                                n_trials=count, balanced_accuracy=.5 + offset +
                                (effect if strategy == 'tucker_learned' and condition['name'] != 'full_22' else 0),
                                log_loss=1.0))
                self.assertAlmostEqual(task_contrasts(cells)[0]['gain'], effect)
                tasks.append(dict(subject=subject, seed=seed, synthetic=False, cells=cells))
        report = aggregate(tasks, [1, 2], [0, 1], complete=True)
        self.assertAlmostEqual(report['contrasts'][0]['mean_gain'], -.05)
        self.assertEqual(report['contrasts'][0]['positive_participants'], 1)
        self.assertAlmostEqual(report['participant_contrasts'][0]['gain'], .15)
        self.assertAlmostEqual(report['participant_contrasts'][1]['gain'], -.25)
        partial = aggregate(tasks[:-1], [1, 2], [0, 1], complete=True)
        self.assertTrue(all(r['mean_gain'] is None and r['ci_95'] is None for r in partial['contrasts']))
        self.assertTrue(all(r['value'] is None for r in partial['cohort_conditions']))
        with self.assertRaisesRegex(ValueError, 'incomplete cell grid'):
            task_contrasts(tasks[0]['cells'][:-1])


if __name__ == '__main__':
    unittest.main()
