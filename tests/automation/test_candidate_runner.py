"""Verify candidate orchestration without real model calls or scientific fits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import test_accuracy_runner as fixtures

ARMS = ['local_control', 'local_power', 'spatial_eegnet',
        'spatial_filterbank', 'spatial_transformer']
CHECKS = ['split_isolation', 'normalization_train_only', 'raw_masking',
          'checkpoint_reload', 'probe_reload', 'mask_pairing', 'identities']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CandidateRunnerTests(fixtures.AccuracyRunnerTests):
    def setUp(self):
        super().setUp()
        for name in ('run_candidate_sessions.sh', 'candidate_session_state.py'):
            shutil.copy2(fixtures.ROOT / 'scripts' / name, self.root / 'scripts' / name)
        old_plan = self.plan
        self.plan = self.root / 'plans/encoder-candidates'
        shutil.copytree(old_plan, self.plan)
        for name in ('COMMON.md', 'CONTRACT.md', 'result.schema.json'):
            shutil.copy2(fixtures.ROOT / 'plans/encoder-candidates' / name, self.plan / name)
        fake = fixtures.FAKE_CODEX.replace('Execute accuracy improvement session',
                                          'Execute encoder candidate session')
        fake = fake.replace('tests/baselines', 'tests/encoder_candidates')
        fake = fake.replace('plans/accuracy/CONTRACT.md', 'plans/encoder-candidates/CONTRACT.md')
        self.fake.write_text(fake)
        self.script = self.root / 'scripts/run_candidate_sessions.sh'
        self.helper = self.root / 'scripts/candidate_session_state.py'

    def completed(self, sid):
        return (self.root / f'.session-runs/encoder-candidates/completed/{sid}.json').is_file()

    def run_helper(self, action, *args):
        return subprocess.run([sys.executable, str(self.helper), action, '--root', str(self.root),
                               '--session', '01', *args], text=True, capture_output=True, timeout=8)

    def add_gate(self):
        path = self.plan / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['sessions'][0]['evidence'] = 'results/encoder-candidates-v1/report/evidence.json'
        save(path, manifest)

    def make_evidence(self):
        self.add_gate()
        config = self.root / 'configs/encoder-candidates.json'
        save(config, {'fixture': 'only tests; no experiment'})
        output = self.root / 'results/encoder-candidates-v1'
        report = {'schema_name': 'agfl-encoder-candidates-report-v1', 'status': 'complete',
                  'synthetic': False, 'partitions': ['train', 'validation'], 'training_regime': 'full',
                  'subjects': list(range(1, 10)), 'seeds': [0, 1, 2], 'arms': ARMS,
                  'study_id': 'a' * 64, 'config_sha256': sha(config),
                  'packages': {'fixture': '1'}, 'source_files_sha256': {'run.py': sha(self.root/'run.py')},
                  'tasks': []}
        shared = ('study_id', 'config_sha256', 'partitions', 'training_regime', 'source_files_sha256')
        for subject in range(1, 10):
            for seed in range(3):
                task = {k: report[k] for k in shared}
                task.update(schema_name='agfl-encoder-candidates-task-v1', status='complete',
                            synthetic=False, subject=subject, seed=seed, split_id='b'*64,
                            data_id='c'*64, checks={k: True for k in CHECKS}, fits=[])
                relative = f'tasks/A{subject:02d}_seed_{seed}/task.json'
                path = output / relative
                for arm in ARMS:
                    fit = {k: task[k] for k in ('subject', 'seed', 'study_id', 'config_sha256',
                                                'split_id', 'data_id', 'partitions')}
                    fit.update(arm=arm, status='complete', synthetic=False, selected_epoch=75,
                               parameter_count=20, clean_train={'balanced_accuracy': .5, 'log_loss': 1.2},
                               validation=[], artifact_sha256={})
                    for condition, repeats in [('full_22', 1), ('random_static_16', 5),
                        ('dynamic_random_16', 5), ('random_static_6', 5), ('dynamic_random_6', 5)]:
                        for repeat in range(repeats):
                            fit['validation'].append({'scenario': condition, 'repeat': repeat,
                                'mask_sha256': hashlib.sha256(f'{condition}-{repeat}'.encode()).hexdigest(),
                                'balanced_accuracy': .4, 'log_loss': 1.4})
                    fit_relative = f'ARMS/{arm}/result.json'
                    fit_path = path.parent / fit_relative
                    names = ['checkpoint.pt', 'history.json', 'predictions.npz']
                    if arm.startswith('local_'):
                        names += ['probe.npz', 'probe_shuffled.npz']
                        fit.update(probe_converged=True, probe_shuffled_converged=True)
                    fit_path.parent.mkdir(parents=True, exist_ok=True)
                    for name in names:
                        artifact = fit_path.parent / name
                        artifact.write_bytes(b'fake scientific data; gate tests only')
                        fit['artifact_sha256'][name] = sha(artifact)
                    save(fit_path, fit)
                    task['fits'].append({'arm': arm, 'path': fit_relative, 'sha256': sha(fit_path)})
                save(path, task)
                report['tasks'].append({'subject': subject, 'seed': seed, 'path': relative, 'sha256': sha(path)})
        self.evidence_path = output / 'report/evidence.json'
        save(self.evidence_path, report)
        return report

    def change_task(self, callback):
        report = json.loads(self.evidence_path.read_text())
        entry = report['tasks'][0]
        path = self.evidence_path.parent.parent / entry['path']
        task = json.loads(path.read_text())
        callback(task, path)
        save(path, task)
        entry['sha256'] = sha(path)
        save(self.evidence_path, report)

    def change_fit(self, callback):
        def update(task, path):
            entry = task['fits'][0]
            fit_path = path.parent / entry['path']
            fit = json.loads(fit_path.read_text())
            callback(fit)
            save(fit_path, fit)
            entry['sha256'] = sha(fit_path)
        self.change_task(update)

    def test_single_session_shorthand_and_conflicting_range(self):
        good = self.run_script('--session', '1')
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual([x['id'] for x in self.calls()], ['01'])
        for args in [('--session', '01', '--through', '02'), ('--from', '01', '--session', '02')]:
            bad = self.run_script(*args)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('not both', bad.stderr)

    def test_repository_default_stops_before_real_evidence_review(self):
        directory = fixtures.ROOT / 'plans/encoder-candidates'
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['default_through'], '09')
        self.assertEqual([s['id'] for s in manifest['sessions']],
                         [f'{i:02d}' for i in range(1, 11)])
        self.assertEqual([s['id'] for s in manifest['sessions'] if 'evidence' in s], ['10'])
        for session in manifest['sessions']:
            self.assertTrue((directory / session['prompt']).is_file())
            self.assertTrue(session['artifacts'])
            self.assertTrue(session['tests'])

    def test_report_blockers_and_empty_checks_are_rejected(self):
        report = {'session_id': '01', 'status': 'completed', 'summary': 'Implemented fixture',
                  'files_changed': [], 'checks': ['Actual fixture check'], 'blockers': [],
                  'next_session_notes': 'Real data unavailable; preflight reports it.'}
        path = self.root/'result.json'
        save(path, report)
        self.assertEqual(self.run_helper('validate-report', '--report', str(path)).returncode, 0)
        for change, message in [({'blockers': ['unfinished implementation']}, 'blockers is nonempty'),
                                ({'checks': []}, 'actual checks'), ({'checks': [' ']}, 'actual checks'),
                                ({'summary': ' '}, 'nonempty summary')]:
            save(path, dict(report, **change))
            result = self.run_helper('validate-report', '--report', str(path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_future_plan_edit_invalidates_receipt(self):
        self.assertEqual(self.run_script('--session', '01').returncode, 0)
        with (self.plan/'02.md').open('a') as stream:
            stream.write('\nChanged future requirement\n')
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale', result.stderr)

    def test_missing_or_synthetic_evidence_prevents_model_call(self):
        self.add_gate()
        missing = self.run_script('--session', '01')
        self.assertNotEqual(missing.returncode, 0)
        self.assertFalse(self.calls())
        report = self.make_evidence()
        report['synthetic'] = True
        save(self.evidence_path, report)
        synthetic = self.run_script('--session', '01')
        self.assertNotEqual(synthetic.returncode, 0)
        self.assertFalse(self.calls())

    def test_complete_evidence_and_cohort_rejections(self):
        report = self.make_evidence()
        valid = self.run_helper('gate')
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertIn('135-fit', valid.stdout)
        for key, value in [('synthetic', 0), ('subjects', [True, *range(2,10)]),
                           ('subjects', [1]), ('seeds', [0]), ('arms', ARMS[:-1]),
                           ('training_regime', 'mixed'), ('partitions', ['train','validation','test']),
                           ('status', 'partial'), ('tasks', report['tasks'][:-1]),
                           ('tasks', [report['tasks'][0]]*27), ('packages', {}),
                           ('study_id', 'missing')]:
            with self.subTest(key=key, value=str(value)[:30]):
                save(self.evidence_path, dict(report, **{key:value}))
                self.assertNotEqual(self.run_helper('gate').returncode, 0)

    def test_config_and_source_changes_rejected(self):
        self.make_evidence()
        config = self.root/'configs/encoder-candidates.json'
        original = config.read_bytes()
        config.write_bytes(original+b' ')
        self.assertIn('config changed', self.run_helper('gate').stderr)
        config.write_bytes(original)
        with (self.root/'run.py').open('a') as stream:
            stream.write('\n# changed\n')
        self.assertIn('source changed', self.run_helper('gate').stderr)

    def test_task_fit_and_numeric_artifact_tampering(self):
        report = self.make_evidence()
        path = self.evidence_path.parent.parent/report['tasks'][0]['path']
        original = path.read_bytes()
        path.write_bytes(original+b' ')
        self.assertIn('Task checksum mismatch', self.run_helper('gate').stderr)
        path.write_bytes(original)
        fit_path = path.parent/'ARMS/local_control/result.json'
        original_fit = fit_path.read_bytes()
        fit_path.write_bytes(original_fit+b' ')
        self.assertIn('Fit checksum mismatch', self.run_helper('gate').stderr)
        fit_path.write_bytes(original_fit)
        (fit_path.parent/'checkpoint.pt').write_bytes(b'changed')
        self.assertIn('artifact checksum mismatch', self.run_helper('gate').stderr)

    def test_task_checks_and_five_unique_arms_required(self):
        self.make_evidence()
        self.change_task(lambda task, path: task['checks'].update(checkpoint_reload=False))
        self.assertIn('Unresolved task checks', self.run_helper('gate').stderr)
        self.change_task(lambda task, path: task['checks'].update(checkpoint_reload=True))
        self.change_task(lambda task, path: task['fits'].__setitem__(1,task['fits'][0]))
        self.assertIn('Duplicate', self.run_helper('gate').stderr)

    def test_pairing_and_probe_failures_rejected(self):
        self.make_evidence()
        self.change_fit(lambda fit: fit.update(probe_converged=False))
        self.assertIn('probe diagnostics', self.run_helper('gate').stderr)
        self.change_fit(lambda fit: fit.update(probe_converged=True))
        self.change_fit(lambda fit: fit['validation'][0].update(mask_sha256='f'*64))
        self.assertIn('not paired', self.run_helper('gate').stderr)

    def test_nonfinite_metrics_and_test_conditions_rejected(self):
        self.make_evidence()
        self.change_fit(lambda fit: fit['clean_train'].update(log_loss=float('nan')))
        self.assertIn('Invalid clean training', self.run_helper('gate').stderr)
        self.change_fit(lambda fit: fit['clean_train'].update(log_loss=1.2))
        self.change_fit(lambda fit: fit['validation'][0].update(scenario='test_full'))
        self.assertIn('Invalid validation condition', self.run_helper('gate').stderr)

    def test_artifact_path_escape_rejected(self):
        self.make_evidence()
        outside = self.root/'outside.npz'
        outside.write_bytes(b'outside')
        self.change_fit(lambda fit: fit['artifact_sha256'].update({'../../../../../../outside.npz':sha(outside)}))
        self.assertIn('inside its output directory', self.run_helper('gate').stderr)

    def test_final_baseline_regression_cannot_skip(self):
        manifest_path = self.plan/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['sessions'][0]['baseline_regression'] = True
        save(manifest_path, manifest)
        test = self.root/'tests/baselines/test_regression.py'
        test.parent.mkdir(parents=True)
        test.write_text('import unittest\nclass Regression(unittest.TestCase):\n'
                        " def test_it(self): self.skipTest('unverified')\n")
        result = self.run_script('--session','01')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.completed('01'))


if __name__ == '__main__':
    import unittest
    unittest.main()
