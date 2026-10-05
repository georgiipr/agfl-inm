"""Follow-up orchestration and explicit experiment commands, with fake execution."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import importlib.util
from unittest import mock

import test_candidate_runner as candidates
import test_accuracy_runner as fixtures


class FollowupRunnerTests(candidates.CandidateRunnerTests):
    def setUp(self):
        super().setUp()
        for name in ('run_followup_sessions.sh', 'followup_session_state.py', 'run_candidate_experiments.sh'):
            shutil.copy2(fixtures.ROOT/'scripts'/name, self.root/'scripts'/name)
        old = self.plan
        self.plan = self.root/'plans/candidate-followup'
        shutil.copytree(old, self.plan)
        for name in ('COMMON.md','CONTRACT.md','result.schema.json'):
            shutil.copy2(fixtures.ROOT/'plans/candidate-followup'/name, self.plan/name)
        self.fake.write_text(self.fake.read_text().replace('Execute encoder candidate session',
                                                          'Execute candidate follow-up session'))
        self.fake.write_text(self.fake.read_text().replace('plans/encoder-candidates/CONTRACT.md',
                                                          'plans/candidate-followup/CONTRACT.md'))
        self.script = self.root/'scripts/run_followup_sessions.sh'
        self.helper = self.root/'scripts/followup_session_state.py'

    def completed(self, sid):
        return (self.root/f'.session-runs/candidate-followup/completed/{sid}.json').is_file()

    def add_gate(self):
        path = self.plan/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['sessions'][0]['evidence'] = 'results/encoder-candidates-reproducible-v1/report/evidence.json'
        candidates.save(path, manifest)

    def make_evidence(self):
        report = super().make_evidence()
        old = self.evidence_path.parent.parent
        new = self.root/'results/encoder-candidates-reproducible-v1'
        old.rename(new)
        self.evidence_path = new/'report/evidence.json'
        (self.root/'configs/encoder-candidates-reproducible.json').symlink_to('encoder-candidates.json')
        for entry in report['tasks']:
            path = new / entry['path']
            task = json.loads(path.read_text())
            task['packages'] = report['packages']
            candidates.save(path, task)
            entry['sha256'] = candidates.sha(path)
        candidates.save(self.evidence_path, report)
        return report

    def test_repository_default_stops_before_real_evidence_review(self):
        directory = fixtures.ROOT/'plans/candidate-followup'
        manifest = json.loads((directory/'manifest.json').read_text())
        self.assertEqual(manifest['default_through'], '02')
        self.assertEqual([s['id'] for s in manifest['sessions']], ['01','02','03','04'])
        self.assertEqual([s['id'] for s in manifest['sessions'] if 'evidence' in s], ['03','04'])
        self.assertTrue(manifest['sessions'][2]['pilot'])
        for session in manifest['sessions']:
            self.assertTrue((directory/session['prompt']).is_file())
            self.assertTrue(session['artifacts'])
            self.assertTrue(session['tests'])

    def test_pilot_gate_needs_only_one_complete_real_task_not_a_cohort_report(self):
        self.make_evidence()
        manifest_path = self.plan/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['sessions'][0]['pilot'] = True
        candidates.save(manifest_path, manifest)
        self.evidence_path.unlink()  # pilot validation must not require a full report
        for directory in (self.evidence_path.parent.parent/'tasks').iterdir():
            if directory.name != 'A01_seed_0':
                shutil.rmtree(directory)
        valid = self.run_helper('gate')
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertIn('1-task/5-fit',valid.stdout)
        path = self.evidence_path.parent.parent/'tasks/A01_seed_0/task.json'
        task = json.loads(path.read_text())
        task['synthetic'] = True
        candidates.save(path,task)
        self.assertNotEqual(self.run_helper('gate').returncode,0)

    def run_experiment(self,*args,**env):
        return subprocess.run(['bash',str(self.root/'scripts/run_candidate_experiments.sh'),*args],
                              cwd='/tmp',env=dict(self.env,**env),text=True,capture_output=True,timeout=10)

    def scientific_calls(self):
        path=self.root/'scientific_calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def fake_science(self):
        # Gates are independently tested above; this isolates Bash ordering/failure handling.
        self.helper.write_text("import os,sys\nif os.environ.get('FAKE_GATE_STOP'): sys.exit(7)\n")
        pkg=self.root/'inm/encoder_candidates'
        pkg.mkdir(parents=True)
        (pkg.parent/'__init__.py').write_text('')
        (pkg/'__init__.py').write_text('')
        (pkg/'__main__.py').write_text('''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
task=int(args[args.index('--task-index')+1]) if '--task-index' in args else 'summary'
with Path('scientific_calls.jsonl').open('a') as f: f.write(json.dumps(task)+'\\n')
if str(task)==os.environ.get('FAKE_FAIL_TASK'): sys.exit(9)
print('fixture execution complete')
''')

    def test_experiment_default_and_dry_run_never_train(self):
        for args in [(),('--help',),('--pilot','--dry-run'),('--cohort','--dry-run'),('--summarize','--dry-run')]:
            result=self.run_experiment(*args)
            self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(self.scientific_calls())
        self.assertFalse((self.root/'.session-runs').exists())
        preview=self.run_experiment('--cohort','--dry-run').stdout
        self.assertEqual(preview.count('--task-index'),26)
        self.assertIn('--summarize-only',preview)

    def test_experiment_without_readiness_stops_before_training(self):
        result=self.run_experiment('--pilot')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.scientific_calls())

    def test_experiment_modes_are_explicit(self):
        for args in [('--pilot','--cohort'),('--device','unknown','--pilot'),('--pilot','--python')]:
            self.assertNotEqual(self.run_experiment(*args).returncode,0)
        self.assertFalse(self.scientific_calls())

    def test_experiment_pilot_only_then_cohort_order_and_summary(self):
        self.fake_science()
        pilot=self.run_experiment('--pilot')
        self.assertEqual(pilot.returncode,0,pilot.stderr)
        self.assertEqual(self.scientific_calls(),[0])
        cohort=self.run_experiment('--cohort')
        self.assertEqual(cohort.returncode,0,cohort.stderr)
        self.assertEqual(self.scientific_calls(),[*range(27),'summary'])
        self.assertFalse(self.calls())

    def test_experiment_failures_stop_without_retry_or_summary(self):
        self.fake_science()
        result=self.run_experiment('--cohort',FAKE_FAIL_TASK='2')
        self.assertEqual(result.returncode,9,result.stderr)
        self.assertEqual(self.scientific_calls(),[1,2])

    def test_experiment_review_gate_stops_calls(self):
        self.fake_science()
        result=self.run_experiment('--cohort',FAKE_GATE_STOP='1')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.scientific_calls())

    def test_experiment_uses_same_workspace_lock(self):
        self.fake_science()
        directory=self.root/'.session-runs'
        directory.mkdir()
        with (directory/'accuracy.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=self.run_experiment('--pilot')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('using this workspace',result.stderr)
        self.assertFalse(self.scientific_calls())

    def test_execution_gate_accepts_existing_output_with_real_config_loader(self):
        spec = importlib.util.spec_from_file_location('followup_real_loader', self.helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        pkg = self.root/'inm/encoder_candidates'
        pkg.mkdir(parents=True)
        shutil.copy2(fixtures.ROOT/'inm/encoder_candidates/protocol.py', pkg/'protocol.py')
        spec = importlib.util.spec_from_file_location('candidate_real_loader', pkg/'protocol.py')
        protocol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol)
        config = self.root/'configs/encoder-candidates-reproducible.json'
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixtures.ROOT/'configs/encoder-candidates-reproducible.json', config)
        output = self.root/'results/encoder-candidates-reproducible-v1'
        output.mkdir(parents=True)
        artifact = output/'existing-artifact.json'
        artifact.write_text('{"preserve": true}\n')
        before = artifact.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already contains artifacts'):
            protocol.load_config(config)
        identity = protocol.study_identity(protocol.load_config(config, allow_existing_output=True))
        directory = self.root/'docs/candidate-followup'
        log = directory/'validation/checks.log'
        log.parent.mkdir(parents=True)
        log.write_text('fixture verification log')
        ready = {**identity, 'status': 'ready', 'verified_devices': ['cpu'],
                 'checks': {k: True for k in ('rng_independence', 'repeat_fit_equivalence',
                                             'all_five_smoke', 'pilot_input_verified')},
                 'artifact_sha256': {'validation/checks.log': candidates.sha(log)}}
        candidates.save(directory/'readiness.json', ready)
        state = self.root/'.session-runs/candidate-followup'
        with mock.patch.object(helper, 'check_receipt'):
            for phase in ('pilot', 'summarize'):
                helper.execution_gate(self.root, state, phase, 'cpu')
            # Cohort reaches its pilot-artifact gate instead of rejecting output.
            with mock.patch.object(helper, 'gate', side_effect=ValueError('pilot sentinel')) as gate:
                with self.assertRaisesRegex(ValueError, 'pilot sentinel'):
                    helper.execution_gate(self.root, state, 'cohort', 'cpu')
                gate.assert_called_once_with(self.root, '03')
            config.write_text(config.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Readiness is stale'):
                helper.execution_gate(self.root, state, 'pilot', 'cpu')
            self.assertEqual(artifact.read_bytes(), before)
            self.assertEqual(list(output.iterdir()), [artifact])
            artifact.unlink()
            output.rmdir()
            output.write_text('not a directory')
            with self.assertRaisesRegex(ValueError, 'not a directory'):
                helper.execution_gate(self.root, state, 'pilot', 'cpu')

    def test_execution_readiness_identity_logs_devices_and_pilot_decision(self):
        # Isolate readiness logic from the independently tested receipt/artifact gate.
        spec = importlib.util.spec_from_file_location('followup_fixture', self.helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        pkg = self.root/'inm/encoder_candidates'
        pkg.mkdir(parents=True)
        (pkg/'protocol.py').write_text('''import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load_config(path, *, allow_existing_output=False):
    assert allow_existing_output, 'execution must inspect existing output'
    return json.loads(Path(path).read_text())
def study_identity(cfg): return json.loads((ROOT/'identity.json').read_text())
''')
        output = self.root/'results/encoder-candidates-reproducible-v1'
        candidates.save(self.root/'configs/encoder-candidates-reproducible.json',
            {'synthetic':False,'subjects':list(range(1,10)),'seeds':[0,1,2],
             'output_dir':str(output)})
        identity = {'study_id':'a'*64,'config_sha256':'b'*64,
                    'source_files_sha256':{'fixture.py':'c'*64},'packages':{'fixture':'1'}}
        candidates.save(self.root/'identity.json',identity)
        directory=self.root/'docs/candidate-followup'
        log=directory/'validation/checks.log'
        log.parent.mkdir(parents=True)
        log.write_text('executed fixture checks')
        ready={**identity,'status':'ready','verified_devices':['cpu'],
               'checks':{k:True for k in ('rng_independence','repeat_fit_equivalence',
                                         'all_five_smoke','pilot_input_verified')},
               'artifact_sha256':{'validation/checks.log':candidates.sha(log)}}
        path=directory/'readiness.json'
        candidates.save(path,ready)
        state=self.root/'.session-runs/candidate-followup'
        with mock.patch.object(helper,'check_receipt'), mock.patch.object(helper,'gate'):
            helper.execution_gate(self.root,state,'pilot','cpu')
            with self.assertRaisesRegex(ValueError,'requested-device'):
                helper.execution_gate(self.root,state,'pilot','cuda')
            candidates.save(self.root/'identity.json',{**identity,'config_sha256':'d'*64})
            with self.assertRaisesRegex(ValueError,'stale'):
                helper.execution_gate(self.root,state,'pilot','cpu')
            candidates.save(self.root/'identity.json',identity)
            log.write_text('tampered')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                helper.execution_gate(self.root,state,'pilot','cpu')
            log.write_text('executed fixture checks')
            task_path=output/'tasks/A01_seed_0/task.json'
            fit_path=task_path.parent/'ARMS/local_control/result.json'
            candidates.save(fit_path,{'execution_device':'cpu'})
            candidates.save(task_path,{'fits':[{'path':'ARMS/local_control/result.json'}]})
            review={'decision':'stop','blockers':['operational issue'],'summary':'Reviewed fixture',
                    'study_id':identity['study_id'],'device':'cpu','pilot_sha256':candidates.sha(task_path)}
            review_path=directory/'pilot-review.json'
            candidates.save(review_path,review)
            with self.assertRaisesRegex(ValueError,'pilot review'):
                helper.execution_gate(self.root,state,'cohort','cpu')
            review.update(decision='proceed',blockers=[])
            candidates.save(review_path,review)
            helper.execution_gate(self.root,state,'cohort','cpu')
            candidates.save(fit_path,{'execution_device':'cuda'})
            with self.assertRaisesRegex(ValueError,'same verified execution device'):
                helper.execution_gate(self.root,state,'cohort','cpu')


if __name__=='__main__':
    import unittest
    unittest.main()
