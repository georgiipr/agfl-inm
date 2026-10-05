"""Strict candidate evidence validation and paired report fixtures."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inm.encoder_candidates.protocol import _package_versions, _source_hashes, arms, load_config, study_identity, tasks
from inm.encoder_candidates.reporting import _bootstrap, summarize


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def metrics(ba=.5, loss=1.0):
    return {"balanced_accuracy": ba, "accuracy": ba, "macro_f1": ba,
            "log_loss": loss, "per_class_recall": [ba] * 4,
            "confusion_matrix": [[1, 0, 0, 0]] * 4}


class CandidateReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = copy.deepcopy(load_config("configs/encoder-candidates.json"))
        self.cfg["output_dir"] = str(self.root / "study")
        self.cfg["data_dir"] = str(self.root / "data")
        self.cfg["split_source_dir"] = str(self.root / "splits")
        self.cfg["subjects"] = list(range(1, 10))
        self.cfg["seeds"] = [0, 1, 2]
        self.out = Path(self.cfg["output_dir"])
        self.study_id = study_identity(self.cfg)["study_id"]
        self.config_id = hashlib.sha256(Path(self.cfg["config_path"]).read_bytes()).hexdigest()
        self.source_map = _source_hashes()
        self.packages = _package_versions()
        self._write_cohort()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_cohort(self):
        for task_spec in tasks(self.cfg):
            subject, seed = task_spec["subject"], task_spec["seed"]
            task_dir = self.out / "tasks" / f"A{subject:02d}_seed_{seed}"
            ids = [f"A{subject:02d}:s{seed}:v{i}" for i in range(3 + subject)]
            mask_values = [("full_22", 0)] + [(scenario, repeat)
                for scenario in ("random_static_16", "dynamic_random_16", "random_static_6", "dynamic_random_6")
                for repeat in range(5)]
            shared_masks = [sha(f"{subject}-{seed}-{scenario}-{repeat}") for scenario, repeat in mask_values]
            fit_rows = []
            for arm in arms(self.cfg):
                arm_dir = task_dir / "ARMS" / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                for name in ("checkpoint.pt",):
                    (arm_dir / name).write_bytes(b"synthetic-fixture-checkpoint")
                history = [{"epoch": 1, "selected": True, "learning_rate": .001,
                    "optimization_loss": 1.0, "optimization_accuracy": .5,
                    "clean_train_metrics": metrics(.6),
                    "clean_validation_metrics": metrics(.5 + (0.03 if arm == "local_power" else 0.0))}]
                (arm_dir / "history.json").write_text(json.dumps(history))
                np.savez(arm_dir / "predictions.npz", validation_sample_ids=np.asarray(ids, dtype=np.str_),
                         validation_labels=np.zeros(len(ids), dtype=np.int64),
                         validation_probabilities=np.full((len(ids), 4), .25))
                if arm.startswith("local_"):
                    for name in ("probe.npz", "probe_shuffled.npz"):
                        np.savez(arm_dir / name, numeric=np.array([1.0]),
                                 train_sample_ids=np.asarray([f"A{subject:02d}:s{seed}:t{i}" for i in range(4)], dtype=np.str_),
                                 validation_sample_ids=np.asarray(ids, dtype=np.str_),
                                 converged=np.asarray(True, dtype=np.bool_))
                is_local = arm.startswith("local_")
                hashes = {name: hashlib.sha256((arm_dir / name).read_bytes()).hexdigest()
                          for name in ["checkpoint.pt", "history.json", "predictions.npz"] +
                          (["probe.npz", "probe_shuffled.npz"] if is_local else [])}
                ba = .5 + (.03 if arm == "local_power" else 0.0)
                validation = [{"scenario": scenario, "repeat": repeat,
                    "mask_sha256": shared_masks[index], "balanced_accuracy": ba,
                    "log_loss": 1.0} for index, (scenario, repeat) in enumerate(mask_values)]
                fit = {"status": "complete", "synthetic": False,
                    "partitions": ["train", "validation"], "training_regime": "full",
                    "subject": subject, "seed": seed, "arm": arm,
                    "study_id": self.study_id, "config_sha256": self.config_id,
                    "split_id": sha(f"split-{subject}-{seed}"), "data_id": sha(f"data-{subject}"),
                    "recording_sha256": {f"A{subject:02d}T.gdf": sha(f"recording-{subject}")},
                    "original_metadata_sha256": {"split.json": sha(f"split-meta-{subject}-{seed}"),
                        "dataset.json": sha(f"dataset-meta-{subject}"), "study.json": sha("study-meta")},
                    "selected_epoch": 1, "parameter_count": 100 + arms(self.cfg).index(arm),
                    "clean_train": metrics(.6), "clean_validation": metrics(ba),
                    "validation": validation, "artifact_sha256": hashes}
                if is_local:
                    fit["probe_converged"] = True
                    fit["probe_shuffled_converged"] = True
                    fit["probe_metrics"] = {kind: {part: metrics(.4) for part in ("train", "validation")}
                                            for kind in ("ordered", "shuffled")}
                result_path = arm_dir / "result.json"
                result_path.write_text(json.dumps(fit))
                fit_rows.append({"arm": arm, "path": f"ARMS/{arm}/result.json",
                                 "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})
            manifest = {"schema_name": "agfl-encoder-candidates-task-v1", "status": "complete",
                "synthetic": False, "partitions": ["train", "validation"], "training_regime": "full",
                "study_id": self.study_id, "config_sha256": self.config_id,
                "source_files_sha256": self.source_map, "packages": self.packages,
                "recording_sha256": {f"A{subject:02d}T.gdf": sha(f"recording-{subject}")},
                "original_metadata_sha256": {"split.json": sha(f"split-meta-{subject}-{seed}"),
                    "dataset.json": sha(f"dataset-meta-{subject}"), "study.json": sha("study-meta")},
                "subject": subject, "seed": seed, "split_id": sha(f"split-{subject}-{seed}"),
                "data_id": sha(f"data-{subject}"), "checks": {key: True for key in (
                    "split_isolation", "normalization_train_only", "raw_masking", "checkpoint_reload",
                    "probe_reload", "mask_pairing", "identities")},
                "partition_sample_ids": {"train": [f"A{subject:02d}:s{seed}:t{i}" for i in range(4)],
                                         "validation": ids}, "fits": fit_rows}
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.json").write_text(json.dumps(manifest))

    def _result_path(self, subject=1, seed=0, arm="local_control"):
        return self.out / "tasks" / f"A{subject:02d}_seed_{seed}" / "ARMS" / arm / "result.json"

    def _update_fit_hash(self, subject, seed, arm):
        directory = self.out / "tasks" / f"A{subject:02d}_seed_{seed}"
        task_path = directory / "task.json"
        manifest = json.loads(task_path.read_text())
        row = next(row for row in manifest["fits"] if row["arm"] == arm)
        result_path = directory / row["path"]
        row["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        task_path.write_text(json.dumps(manifest))

    def test_hierarchical_paired_report_ignores_trial_count_and_preserves_all_scores(self):
        result = summarize(self.cfg)
        self.assertTrue(result["complete"])
        report = self.out / "report"
        with (report / "validation_contrasts.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        gain = next(row for row in rows if row["scope"] == "cohort" and row["metric"] == "balanced_accuracy")
        self.assertAlmostEqual(float(gain["gain"]), .03)
        self.assertEqual(gain["positive_participants"], "9")
        self.assertEqual(gain["promising_for_confirmation"], "True")
        with (report / "per_subject_scores.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 135)
        evidence = json.loads((report / "evidence.json").read_text())
        self.assertEqual(evidence["status"], "complete")
        self.assertEqual(len(evidence["tasks"]), 27)
        # Participant bootstrap resamples are deterministic and paired values retain the exact CI.
        interval = _bootstrap([.03] * 9, 2000, 20261003)
        self.assertAlmostEqual(interval[0], .03)
        self.assertAlmostEqual(interval[1], .03)

    def test_missing_failed_duplicate_synthetic_and_partial_evidence_withhold_cohort(self):
        task_path = self.out / "tasks/A01_seed_0/task.json"
        manifest = json.loads(task_path.read_text())
        manifest["fits"] = manifest["fits"][:-1]
        task_path.write_text(json.dumps(manifest))
        result = summarize(self.cfg)
        self.assertFalse(result["complete"])
        self.assertEqual(json.loads((self.out / "report/evidence.json").read_text())["tasks"], [])
        self.assertTrue(result["issues"])
        failed_path = self.out / "tasks/A02_seed_0/task.json"
        failed = json.loads(failed_path.read_text())
        failed["status"] = "failed"
        failed_path.write_text(json.dumps(failed))
        result = summarize(self.cfg)
        self.assertTrue(any(issue["subject"] == 2 and "failed or synthetic" in issue["reason"]
                            for issue in result["issues"]))
        # Synthetic input cannot become eligible, even with otherwise complete checks.
        manifest["fits"] = json.loads((self.out / "tasks/A01_seed_0/task.json").read_text()).get("fits", [])
        manifest["synthetic"] = True
        task_path.write_text(json.dumps(manifest))
        result = summarize(self.cfg)
        self.assertFalse(result["complete"])
        self.assertIn("synthetic", result["issues"][0]["reason"])

    def test_mask_split_and_artifact_tampering_are_rejected(self):
        path = self._result_path(1, 0, "local_power")
        result = json.loads(path.read_text())
        result["validation"][2]["mask_sha256"] = sha("mismatch")
        path.write_text(json.dumps(result))
        self._update_fit_hash(1, 0, "local_power")
        report = summarize(self.cfg)
        self.assertFalse(report["complete"])
        self.assertIn("masks differ", report["issues"][0]["reason"])

        # Corrupt a hashed checkpoint and ensure the independent artifact audit catches it.
        checkpoint = self.out / "tasks/A02_seed_0/ARMS/spatial_eegnet/checkpoint.pt"
        checkpoint.write_bytes(b"tampered")
        report = summarize(self.cfg)
        self.assertFalse(report["complete"])
        self.assertTrue(any("artifact checksum" in issue["reason"] for issue in report["issues"]))

    def test_split_id_mismatch_and_duplicate_arm_are_rejected(self):
        path = self._result_path(1, 0, "spatial_eegnet")
        result = json.loads(path.read_text())
        result["split_id"] = sha("other split")
        path.write_text(json.dumps(result))
        self._update_fit_hash(1, 0, "spatial_eegnet")
        report = summarize(self.cfg)
        self.assertFalse(report["complete"])
        self.assertIn("fit identity mismatch", report["issues"][0]["reason"])

        manifest_path = self.out / "tasks/A02_seed_0/task.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fits"][1]["arm"] = manifest["fits"][0]["arm"]
        manifest_path.write_text(json.dumps(manifest))
        report = summarize(self.cfg)
        self.assertFalse(report["complete"])
        self.assertIn("duplicate, or unordered arms", report["issues"][1]["reason"])

    def test_negative_screening_and_finite_validation(self):
        for subject in range(1, 10):
            for seed in range(3):
                path = self._result_path(subject, seed, "local_power")
                result = json.loads(path.read_text())
                result["clean_validation"] = metrics(.45)
                for row in result["validation"]:
                    row["balanced_accuracy"] = .45
                path.write_text(json.dumps(result))
                history_path = path.parent / "history.json"
                history = json.loads(history_path.read_text())
                history[0]["clean_validation_metrics"] = metrics(.45)
                history_path.write_text(json.dumps(history))
                result = json.loads(path.read_text())
                result["artifact_sha256"]["history.json"] = hashlib.sha256(history_path.read_bytes()).hexdigest()
                path.write_text(json.dumps(result))
                self._update_fit_hash(subject, seed, "local_power")
        report = summarize(self.cfg)
        self.assertTrue(report["complete"])
        with (self.out / "report/validation_contrasts.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        primary = next(row for row in rows if row["scope"] == "cohort" and row["metric"] == "balanced_accuracy")
        self.assertLess(float(primary["gain"]), 0)
        self.assertEqual(primary["promising_for_confirmation"], "False")

    def test_nonfinite_metrics_cannot_enter_report(self):
        path = self._result_path(1, 0, "local_control")
        result = json.loads(path.read_text())
        result["validation"][0]["balanced_accuracy"] = float("nan")
        path.write_text(json.dumps(result))
        self._update_fit_hash(1, 0, "local_control")
        summary = summarize(self.cfg)
        self.assertFalse(summary["complete"])
        self.assertIn("finite number", summary["issues"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
