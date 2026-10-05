"""Independent synthetic acceptance checks for the supervised Tucker comparison."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from inm.encoder_candidates.data import PartitionData, PreparedData
from inm.supervised_tucker import study


ROOT = Path(__file__).resolve().parents[2]


def prepared(seed=0):
    """Distinct train/validation IDs; no test attribute exists."""
    rng = np.random.default_rng(182)
    labels = np.tile(np.arange(4, dtype=np.int64), 2)
    time = np.arange(250) / 250
    def partition(prefix):
        raw = rng.normal(0, .2, (8, 22, 4, 250)).astype(np.float32)
        for index, label in enumerate(labels):
            raw[index, label::4] += np.sin(2 * np.pi * (8 + 4 * label) * time)
        return PartitionData(raw, labels.copy(), tuple(f"{prefix}-{i}" for i in range(8)))
    return PreparedData(1, seed, partition("train"), partition("validation"),
                        {"mean": [0.] * 22, "std": [1.] * 22},
                        "synthetic-split", "synthetic-data", {"synthetic": True})


def config(directory, *, name="synthetic-screen"):
    path = ROOT / "configs" / "supervised-tucker.json"
    value = json.loads(path.read_text())
    for key in ("data_dir", "split_source_dir"):
        value[key] = str((path.parent / value[key]).resolve())
    value.update(synthetic=True, subjects=[1], seeds=[0], name=name,
                 calibration_epochs=1, output_dir=str(Path(directory) / name))
    value["training"].update(maximum_epochs=2, minimum_epochs=1,
                             patience=1, batch_size=8, warmup_epochs=1)
    destination = Path(directory) / f"{name}.json"
    destination.write_text(json.dumps(value))
    return study.load_config(destination)


class StudyAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_arms_finish_and_reuse_without_rewriting(self):
        data = prepared()
        self.assertFalse(hasattr(data, "test"))
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            for arm in cfg["arms"]:
                with self.subTest(arm=arm):
                    destination = Path(temporary) / arm
                    result = study.fit_arm(data, cfg, arm, destination)
                    paths = [destination / name for name in (
                        "checkpoint.pt", "history.json", "predictions.npz",
                        "robustness.json", "result.json")]
                    before = {path.name: path.read_bytes() for path in paths}
                    self.assertTrue(all(path.is_file() for path in paths))
                    self.assertTrue(np.isfinite(result["validation_metrics"]["balanced_accuracy"]))
                    self.assertIn(result["selected_epoch"], (1, 2))
                    history = json.loads((destination / "history.json").read_text())
                    expected = max(history, key=lambda row: (
                        row["clean_validation_metrics"]["balanced_accuracy"],
                        -row["clean_validation_metrics"]["log_loss"], -row["epoch"]))
                    self.assertEqual(result["selected_epoch"], expected["epoch"])
                    with np.load(destination / "predictions.npz", allow_pickle=False) as saved:
                        self.assertEqual(set(saved.files), {
                            "train_labels", "validation_labels", "train_probabilities",
                            "validation_probabilities", "train_sample_ids", "validation_sample_ids"})
                        predicted = saved["validation_probabilities"].argmax(axis=1)
                        labels = saved["validation_labels"]
                        balanced_accuracy = np.mean([
                            np.mean(predicted[labels == label] == label) for label in range(4)])
                        self.assertEqual(result["validation_metrics"]["balanced_accuracy"], balanced_accuracy)
                    again = study.fit_arm(data, cfg, arm, destination)
                    self.assertEqual(result, again)
                    self.assertEqual(before, {path.name: path.read_bytes() for path in paths})

    def test_corrupt_selected_checkpoint_is_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            destination = Path(temporary) / "corrupt"
            study.fit_arm(prepared(), cfg, "spectral_tucker_frozen", destination)
            checkpoint = destination / "checkpoint.pt"
            checkpoint.write_bytes(b"corrupted checkpoint")
            with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
                study.fit_arm(prepared(), cfg, "spectral_tucker_frozen", destination)
            self.assertEqual(checkpoint.read_bytes(), b"corrupted checkpoint")

    def test_unrecognized_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            destination = Path(temporary) / "occupied"
            destination.mkdir()
            sentinel = destination / "unrelated.txt"
            sentinel.write_text("preserve me")
            with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
                study.fit_arm(prepared(), cfg, "spectral_tucker_frozen", destination)
            self.assertEqual(sentinel.read_text(), "preserve me")

    def test_changed_data_identity_is_not_reused(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            destination = Path(temporary) / "identity"
            data = prepared()
            study.fit_arm(data, cfg, "spectral_tucker_frozen", destination)
            with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
                study.fit_arm(replace(data, data_id="different"), cfg,
                              "spectral_tucker_frozen", destination)

    def test_real_config_rejects_shortened_training_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            config(temporary)
            path = Path(temporary) / "synthetic-screen.json"
            value = json.loads(path.read_text())
            value["synthetic"] = False
            value["output_dir"] = str(Path(temporary) / "real-screen")
            path.write_text(json.dumps(value))
            with self.assertRaises((ValueError, RuntimeError)):
                study.load_config(path)

    def test_two_fits_are_identical_despite_external_random_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            data = prepared()
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            study.fit_arm(data, cfg, "spectral_tucker_supervised", first)
            torch.rand(731)
            np.random.normal(size=123)
            study.fit_arm(data, cfg, "spectral_tucker_supervised", second)
            self.assertEqual((first / "history.json").read_bytes(),
                             (second / "history.json").read_bytes())
            with np.load(first / "predictions.npz") as left, np.load(second / "predictions.npz") as right:
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])
            left = torch.load(first / "checkpoint.pt", weights_only=True)
            right = torch.load(second / "checkpoint.pt", weights_only=True)
            for key in left["state_dict"]:
                torch.testing.assert_close(left["state_dict"][key], right["state_dict"][key],
                                           rtol=0, atol=0)

    def test_changed_config_cannot_resume_study_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            study.ensure_study(cfg)
            path = Path(cfg["config_path"])
            value = json.loads(path.read_text())
            value["name"] = "changed scientifically declared identity"
            path.write_text(json.dumps(value))
            changed = study.load_config(path)
            with self.assertRaisesRegex(ValueError, "changed|identity"):
                study.ensure_study(changed)

    def test_partial_report_has_no_cohort_mean_and_completed_masks_are_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            path = Path(cfg["config_path"])
            value = json.loads(path.read_text())
            value["subjects"] = [1, 2]
            path.write_text(json.dumps(value))
            cfg = study.load_config(path)
            study.run_task(cfg, 0, synthetic=True)
            partial = study.summarize(cfg)
            self.assertEqual(partial["status"], "partial")
            self.assertIs(partial["synthetic"], True)
            self.assertEqual(partial["cohort"], [])
            self.assertEqual(partial["contrasts"], [])
            self.assertEqual(len(partial["rows"]), 2)
            study.run_task(cfg, 1, synthetic=True)
            complete = study.summarize(cfg)
            self.assertEqual(complete["status"], "complete")
            self.assertIs(complete["synthetic"], True)
            self.assertEqual(len(complete["cohort"]), 2)
            self.assertEqual(len(complete["rows"]), 4)
            for aggregate in complete["cohort"]:
                scores = [row["validation_balanced_accuracy"] for row in complete["rows"]
                          if row["arm"] == aggregate["arm"]]
                self.assertEqual(aggregate["validation_balanced_accuracy"], np.mean(scores))
            for subject in (1, 2):
                mask_banks = []
                for arm in cfg["arms"]:
                    rows = json.loads((study.task_directory(cfg, subject) / arm / "robustness.json").read_text())
                    self.assertEqual(len(rows), 21)
                    mask_banks.append([(row["scenario"], row["repeat"], row["mask_sha256"]) for row in rows])
                self.assertTrue(all(bank == mask_banks[0] for bank in mask_banks))

    def test_overlapping_train_validation_ids_fail_before_artifacts(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            data = prepared()
            data = replace(data, validation=replace(data.validation, sample_ids=data.train.sample_ids))
            destination = Path(temporary) / "overlap"
            with self.assertRaisesRegex(ValueError, "overlap"):
                study.fit_arm(data, cfg, "spectral_tucker_frozen", destination)
            self.assertFalse(destination.exists())

    def test_selection_prioritizes_accuracy_then_loss_then_earliest_epoch(self):
        from inm.encoder_candidates import training
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            path = Path(cfg["config_path"])
            value = json.loads(path.read_text())
            value["training"].update(maximum_epochs=4, minimum_epochs=4)
            path.write_text(json.dumps(value))
            cfg = study.load_config(path)
            scripted = [(0.5, .2), (.75, .8), (.75, .6), (.75, .6)]
            calls = 0
            def evaluate(model, x, labels, batch_size, device):
                nonlocal calls
                call = calls
                calls += 1
                if call % 2 == 0:
                    accuracy, loss = .5, 1.
                else:
                    accuracy, loss = scripted[min(call // 2, 3)]
                return {"balanced_accuracy": accuracy, "log_loss": loss}, np.full((len(labels), 4), .25)
            with patch.object(training, "_evaluate", side_effect=evaluate):
                result = study.fit_arm(prepared(), cfg, "spectral_tucker_frozen", Path(temporary) / "selection")
            self.assertEqual(result["selected_epoch"], 3)
            self.assertEqual(result["epochs_run"], 4)

    def test_pair_calibration_frozen_factor_and_pilot_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            study.run_task(cfg, 0, synthetic=True)
            results = study.verify_task(cfg, 1, 0, study.study_identity(cfg))
            self.assertEqual(results[0]['initial_state_sha256'], results[1]['initial_state_sha256'])
            self.assertEqual(results[0]['calibration_sha256'], results[1]['calibration_sha256'])
            self.assertFalse(results[0]['factors_changed'])
            self.assertTrue(results[1]['factors_changed'])
            self.assertEqual(study.verify_pilot(cfg)['status'], 'passed')
            broken = [dict(result) for result in results]
            broken[1]['initial_state_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'initialization/calibration'):
                study.verify_pair(broken)

    def test_aggregation_averages_repetitions_before_participant_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            cfg['subjects'] = [1, 2]
            cfg['seeds'] = [0, 1, 2]
            rows, robust = [], []
            # Seed2 contradicts the average so keying only by subject must fail.
            values = {1: [.6, .6, .1], 2: [.8, .8, .2]}
            for subject in cfg['subjects']:
                for seed in cfg['seeds']:
                    for arm in study.ARM_IDS:
                        score = .25 if arm == study.ARM_IDS[0] else values[subject][seed]
                        rows.append({'subject': subject, 'seed': seed, 'arm': arm,
                            'train_balanced_accuracy': score + .1,
                            'validation_balanced_accuracy': score, 'validation_accuracy': score,
                            'validation_log_loss': 2. - score, 'train_validation_gap': .1})
                        for condition in cfg['conditions']:
                            robust.append({'subject': subject, 'seed': seed, 'arm': arm,
                                'scenario': condition['name'], 'balanced_accuracy': score, 'log_loss': 2.-score})
            report = study.aggregate_complete(cfg, rows, robust)
            mean = np.mean([np.mean(values[1]), np.mean(values[2])])
            self.assertAlmostEqual(report['cohort'][1]['validation_balanced_accuracy'], mean)
            self.assertEqual(report['cohort'][1]['participants'], 2)
            self.assertEqual(report['cohort'][1]['seeds_per_participant'], 3)
            primary = report['contrasts'][0]
            self.assertAlmostEqual(primary['mean_difference'], mean - .25)
            self.assertEqual(primary['positive_participants'], 2)
            self.assertEqual(len(primary['participant_differences']), 2)
            self.assertEqual(len(report['seed_contrasts']), 12)
            self.assertTrue(all(value >= 0 for value in primary['ci95_participant_bootstrap']))
            self.assertAlmostEqual(report['robustness_cohort'][5]['balanced_accuracy'], mean)
            missing = study.aggregate_complete(cfg, rows[:-1], robust)
            self.assertTrue(all(not value for value in missing.values()))
            missing_robust = study.aggregate_complete(cfg, rows, robust[:-1])
            self.assertTrue(all(not value for value in missing_robust.values()))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                study.aggregate_complete(cfg, rows + rows[:1], robust)

    def test_missing_seed_suppresses_real_report_aggregates(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            path = Path(cfg['config_path'])
            value = json.loads(path.read_text())
            value['seeds'] = [0, 1, 2]
            path.write_text(json.dumps(value))
            cfg = study.load_config(path)
            study.run_task(cfg, 0, synthetic=True)
            report = study.summarize(cfg)
            self.assertEqual(report['status'], 'partial')
            self.assertEqual({item['seed'] for item in report['missing']}, {1, 2})
            for name in ('cohort', 'contrasts', 'participant_scores', 'robustness_cohort', 'robustness_contrasts'):
                self.assertEqual(report[name], [])

    def test_partial_fit_cannot_resume_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            destination = Path(temporary) / 'partial'
            destination.mkdir()
            sentinel = destination / 'in_progress.json'
            sentinel.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                study.fit_arm(prepared(), cfg, study.ARM_IDS[0], destination)
            self.assertEqual(sentinel.read_text(), '{}')

    def test_real_config_requires_three_seeds_and_unknown_keys_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = json.loads((ROOT / 'configs/supervised-tucker.json').read_text())
            value['output_dir'] = str(Path(temporary) / 'real-study')
            path = Path(temporary) / 'real.json'
            value['seeds'] = [0]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'seeds'):
                study.load_config(path)
            value['seeds'] = [0, 1, 2]
            value['tuning'] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'unknown'):
                study.load_config(path)


if __name__ == "__main__":
    unittest.main()
