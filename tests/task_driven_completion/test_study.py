"""Synthetic end-to-end numeric artifacts, failure boundaries and aggregation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from inm.task_driven_completion.protocol import (load_config,study_identity,file_sha256,STRATEGY_IDS,CONDITIONS)
from inm.task_driven_completion.study import (run_task,synthetic_context,_verify_complete_task,
    initialize_output,_atomic_json,_atomic_npz)
from inm.task_driven_completion.reporting import summarize,aggregate

ROOT = Path(__file__).resolve().parents[2]


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        declaration = json.loads((ROOT/'configs/task-driven-completion.json').read_text())
        declaration.update(synthetic=True,subjects=[1],seeds=[0],data_dir='data',candidate_source_dir='candidate',
                           split_source_dir='splits',output_dir='output')
        cls.config_path = cls.root/'config.json'
        cls.config_path.write_text(json.dumps(declaration))
        cls.cfg = load_config(cls.config_path)
        cls.context = synthetic_context()
        cls.result = run_task(cls.cfg,0,synthetic_context=cls.context)
        cls.task_dir = cls.root/'output/tasks/A01_seed_0'
        cls.identity = study_identity(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_five_strategies_numeric_reload_and_resume(self):
        self.assertEqual(len(self.result['cells']),105)
        self.assertEqual(set(self.result['fits']),set(STRATEGY_IDS))
        self.assertTrue(self.result['exact_selected_state_replay'])
        self.assertTrue(self.result['backbone_unchanged'])
        before = {str(p):file_sha256(p) for p in self.task_dir.rglob('*') if p.is_file()}
        self.assertEqual(run_task(self.cfg,0,synthetic_context=synthetic_context()),self.result)
        self.assertEqual(before,{str(p):file_sha256(p) for p in self.task_dir.rglob('*') if p.is_file()})

    def test_changed_budget_is_rejected_on_resume(self):
        context = synthetic_context()
        context['budget']['maximum_epochs'] = 2
        with self.assertRaisesRegex(ValueError,'Changed synthetic'):
            run_task(self.cfg,0,synthetic_context=context)

    def test_numeric_artifact_corruption_and_metric_tampering_rejected(self):
        path = self.task_dir/self.result['cells'][0]['artifact']
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                _verify_complete_task(self.task_dir,self.identity)
        finally:
            path.write_bytes(original)
        task_path = self.task_dir/'task.json'
        original = task_path.read_bytes()
        try:
            task = copy.deepcopy(self.result)
            task['cells'][0]['balanced_accuracy'] += .125
            _atomic_json(task_path,task)
            with self.assertRaisesRegex(ValueError,'Metrics'):
                _verify_complete_task(self.task_dir,self.identity)
        finally:
            task_path.write_bytes(original)

    def test_rehashed_mask_tampering_rejected_by_regeneration(self):
        cell = next(r for r in self.result['cells'] if r['condition']=='random_static_16')
        path = self.task_dir/cell['artifact']; original = path.read_bytes()
        task_path = self.task_dir/'task.json'; original_task = task_path.read_bytes()
        try:
            with np.load(path,allow_pickle=False) as saved:
                arrays = dict(saved)
            arrays['mask'][0,:,0] = np.roll(arrays['mask'][0,:,0],1)
            _atomic_npz(path,**arrays)
            task = copy.deepcopy(self.result); task['artifacts'][cell['artifact']] = file_sha256(path)
            _atomic_json(task_path,task)
            with self.assertRaisesRegex(ValueError,'masks'):
                _verify_complete_task(self.task_dir,self.identity)
        finally:
            path.write_bytes(original); task_path.write_bytes(original_task)

    def test_selected_epoch_and_coverage_tampering_rejected(self):
        task_path = self.task_dir/'task.json'; original = task_path.read_bytes()
        try:
            task = copy.deepcopy(self.result); task['cells'].pop()
            _atomic_json(task_path,task)
            with self.assertRaisesRegex(ValueError,'grid'):
                _verify_complete_task(self.task_dir,self.identity)
            task = copy.deepcopy(self.result); task['fits']['tucker_learned']['selected_epoch'] = 999
            _atomic_json(task_path,task)
            with self.assertRaisesRegex(ValueError,'epoch'):
                _verify_complete_task(self.task_dir,self.identity)
        finally:
            task_path.write_bytes(original)

    def test_rehashed_calibration_history_tampering_rejected(self):
        path = self.task_dir/'factor-history.json'; original = path.read_bytes()
        task_path = self.task_dir/'task.json'; original_task = task_path.read_bytes()
        try:
            history = json.loads(original)
            history[0]['epoch'] = 9
            _atomic_json(path,history)
            task = copy.deepcopy(self.result); task['artifacts']['factor-history.json'] = file_sha256(path)
            _atomic_json(task_path,task)
            with self.assertRaisesRegex(ValueError,'calibration history'):
                _verify_complete_task(self.task_dir,self.identity)
        finally:
            path.write_bytes(original); task_path.write_bytes(original_task)

    def test_current_source_identity_change_rejected(self):
        changed = copy.deepcopy(self.identity); changed['study_id'] = 'different'
        with patch('inm.task_driven_completion.study.study_identity',return_value=changed):
            with self.assertRaisesRegex(ValueError,'Changed source'):
                initialize_output(self.cfg)
        with patch('inm.task_driven_completion.reporting.study_identity',return_value=changed):
            with self.assertRaisesRegex(ValueError,'identity'):
                summarize(self.cfg)

    def test_partial_failed_foreign_output_rejected(self):
        task_path = self.task_dir/'task.json'; original = task_path.read_bytes()
        try:
            task_path.unlink()
            with self.assertRaisesRegex(ValueError,'Partial'):
                run_task(self.cfg,0,synthetic_context=self.context)
            task = copy.deepcopy(self.result); task['status'] = 'failed'
            _atomic_json(task_path,task)
            with self.assertRaisesRegex(ValueError,'Failed'):
                run_task(self.cfg,0,synthetic_context=self.context)
        finally:
            task_path.write_bytes(original)
        with tempfile.TemporaryDirectory() as directory:
            cfg = copy.deepcopy(self.cfg); cfg['output_dir'] = directory
            (Path(directory)/'foreign').write_text('owned elsewhere')
            with self.assertRaisesRegex(ValueError,'Foreign'):
                initialize_output(cfg)

    def test_synthetic_excluded_and_incomplete_means_suppressed(self):
        report = summarize(self.cfg)
        self.assertEqual(report['valid_tasks'],0)
        self.assertEqual(len(report['synthetic_tasks_excluded']),1)
        self.assertFalse(report['complete'])
        self.assertTrue(all(row['mean_gain'] is None for row in report['contrasts']))
        self.assertTrue(all(row['value'] is None for row in report['cohort_conditions']))
        with self.assertRaisesRegex(ValueError,'Synthetic'):
            aggregate([self.result],[1],[0])

    def test_real_replay_failure_precedes_any_calibration(self):
        cfg = copy.deepcopy(self.cfg); cfg['synthetic'] = False
        with tempfile.TemporaryDirectory() as directory:
            cfg['output_dir'] = str(Path(directory)/'output')
            with patch('inm.completion_transformer.replay.prepare_task',return_value=(self.context['prepared'],{})), \
                 patch('inm.completion_transformer.replay.replay_full_input',side_effect=ValueError('original replay failed')), \
                 patch('inm.task_driven_completion.completion.fit_initialization') as calibration:
                with self.assertRaisesRegex(ValueError,'original replay failed'):
                    run_task(cfg,0)
                calibration.assert_not_called()
                status = json.loads((Path(cfg['output_dir'])/'tasks/A01_seed_0/task.json').read_text())
                self.assertEqual(status['status'],'failed')

    def test_hierarchical_aggregation_preserves_negative_participant_effects(self):
        rows = []
        for subject,seeds,gain in ((1,[0,1],.10),(2,[0,1],-.04)):
            for seed in seeds:
                cells = []
                for strategy in STRATEGY_IDS:
                    for condition in CONDITIONS:
                        for repeat in range(condition['repeats']):
                            cells.append(dict(strategy=strategy,condition=condition['name'],repeat=repeat,
                                balanced_accuracy=.5+(gain if strategy=='tucker_learned' else 0),log_loss=1.0))
                rows.append(dict(subject=subject,seed=seed,synthetic=False,cells=cells))
        result = aggregate(rows,[1,2],[0,1],complete=True)
        self.assertAlmostEqual(result['contrasts'][0]['mean_gain'],.03)
        self.assertEqual(result['contrasts'][0]['positive_participants'],1)
        effects = [r['gain'] for r in result['participant_contrasts'] if r['contrast'].startswith('primary')]
        np.testing.assert_allclose(effects,[.1,-.04])
        self.assertEqual(result['contrasts'][0]['ci_95'],aggregate(rows,[1,2],[0,1],complete=True)['contrasts'][0]['ci_95'])
        partial = aggregate(rows,[1,2],[0,1],complete=False)
        self.assertIsNone(partial['contrasts'][0]['mean_gain'])
        # A caller cannot promote incomplete evidence by passing complete=True.
        partial = aggregate(rows[:-1],[1,2],[0,1],complete=True)
        self.assertIsNone(partial['contrasts'][0]['mean_gain'])
        effects = [r['gain'] for r in partial['participant_contrasts'] if r['contrast'].startswith('primary')]
        np.testing.assert_allclose(effects,[.1,-.04])
