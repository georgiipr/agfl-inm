from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from inm.availability import CHANNEL_IDS, make_mask_bank, mask_bank_digest
from inm.completion_transformer.protocol import ARM_IDS, CONDITIONS, STRATEGY_IDS, load_config
from inm.completion_transformer.reporting import summarize, verify_task_artifacts
from inm.completion_transformer.study import _atomic_json, _atomic_npz, _verify_complete_task, initialize_output
from inm.completion_transformer.completion import (CovarianceCompleter, TuckerCompleter,
                                                    completion_state_arrays)
from inm.encoder_candidates.training import _metrics

ROOT = Path(__file__).resolve().parents[2]


class SyntheticArtifactReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = load_config(ROOT / "configs/completion-transformer.json", allow_existing_output=True)
        self.cfg = copy.deepcopy(base)
        self.cfg.update(subjects=[1], seeds=[0], synthetic=True, output_dir=str(Path(self.temp.name) / "out"))
        self.identity = initialize_output(self.cfg)
        self.task_dir = Path(self.cfg["output_dir"]) / "tasks/A01_seed_0"
        self.task_dir.mkdir(parents=True)
        self._build_task()

    def tearDown(self):
        self.temp.cleanup()

    def _build_task(self):
        labels = np.arange(4, dtype=np.int64)
        ids = [f"synthetic:val:{i}" for i in range(4)]
        probabilities = np.full((4, 4), .1, dtype=np.float64)
        probabilities[np.arange(4), labels] = .7
        rows, files = [], {}
        for condition in CONDITIONS:
            for repeat in range(condition["repeats"]):
                mask = make_mask_bank(4, 4, condition["retained"], condition["pattern"],
                    seed=0, partition="validation", subject="A01", repeat=repeat,
                    sample_ids=ids, channel_ids=CHANNEL_IDS)
                mh = mask_bank_digest(mask)
                for backbone in ARM_IDS:
                    for strategy in STRATEGY_IDS:
                        score = _metrics(labels, probabilities)
                        name = f"cells/{backbone}__{strategy}__{condition['name']}__r{repeat}.npz"
                        _atomic_npz(self.task_dir / name, probabilities=probabilities, labels=labels,
                            sample_ids=np.asarray(ids, dtype=np.str_), mask=mask,
                            mask_sha256=np.asarray(mh),
                            metric_balanced_accuracy=np.asarray(score["balanced_accuracy"]),
                            metric_log_loss=np.asarray(score["log_loss"]), subject=np.asarray(1),
                            seed=np.asarray(0), backbone=np.asarray(backbone), strategy=np.asarray(strategy),
                            condition=np.asarray(condition["name"]), repeat=np.asarray(repeat))
                        rows.append({"backbone": backbone, "strategy": strategy, "condition": condition["name"],
                            "repeat": repeat, "mask_sha256": mh, "artifact": name,
                            "balanced_accuracy": score["balanced_accuracy"], "log_loss": score["log_loss"]})
        covariance = CovarianceCompleter().eval()
        covariance.second_moment.copy_(torch.eye(22))
        covariance.fitted.fill_(True)
        covariance.training_samples.fill_(8 * 4 * 250)
        tucker = TuckerCompleter(epochs=1).eval()
        with torch.no_grad():
            tucker.model.U.copy_(torch.eye(22, 4))
            tucker.model.V.copy_(torch.eye(250, 16))
            tucker.model.fitted.fill_(True)
            tucker.model.fit_steps.fill_(1)
            tucker.model.training_seen_counts.fill_(8 * 4)
        _atomic_npz(self.task_dir / "completion-state.npz",
                    **completion_state_arrays(covariance, tucker))
        history = [{"epoch": 1, "objective": 1.0, "reconstruction_mse": .9,
                    "core_penalty": .1}]
        _atomic_json(self.task_dir / "completion-history.json", history)
        task = {"schema_name": "agfl-completion-transformer-task-v1", "status": "complete", "synthetic": True,
            "subject": 1, "seed": 0, "study_id": self.identity["study_id"],
            "config_sha256": self.identity["config_sha256"], "validation_sample_ids": ids,
            "validation_labels": labels.tolist(), "tucker_fit_epochs": 1,
            "synthetic_factor_epoch_override": 1, "cells": rows}
        files = {path.relative_to(self.task_dir).as_posix(): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                 for path in self.task_dir.rglob("*") if path.is_file()}
        task["artifacts"] = files
        _atomic_json(self.task_dir / "task.json", task)
        self.task = task

    def test_numeric_cells_metrics_pairing_and_report_exclusion(self):
        verify_task_artifacts(self.task_dir, self.task)
        verified = _verify_complete_task(self.task_dir, self.identity, synthetic=True)
        self.assertEqual(verified["status"], "complete")
        report = summarize(self.cfg)
        self.assertFalse(report["complete"])
        evidence = json.loads((Path(report["report_dir"]) / "evidence.json").read_text())
        self.assertEqual(evidence["valid_real_tasks"], 0)
        self.assertEqual(len(evidence["synthetic_tasks_excluded"]), 1)
        self.assertIsNone(evidence["contrasts"][0]["mean_gain"])
        self.assertIsNone(evidence["contrasts"][0]["ci_95"])

    def test_corrupt_numeric_artifact_is_rejected(self):
        path = self.task_dir / self.task["cells"][0]["artifact"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            _verify_complete_task(self.task_dir, self.identity, synthetic=True)

    def test_metric_is_recomputed_instead_of_trusting_task_row(self):
        self.task["cells"][0]["balanced_accuracy"] = .25
        with self.assertRaisesRegex(ValueError, "task row metrics"):
            verify_task_artifacts(self.task_dir, self.task)

    def test_corrupt_state_is_rejected_even_after_checksum_recomputed(self):
        path = self.task_dir / "completion-state.npz"
        with np.load(path, allow_pickle=False) as state:
            arrays = {name: state[name].copy() for name in state.files}
        arrays["tucker__U"][0, 0] = np.nan
        _atomic_npz(path, **arrays)
        self.task["artifacts"] = {
            file.relative_to(self.task_dir).as_posix(): __import__("hashlib").sha256(file.read_bytes()).hexdigest()
            for file in self.task_dir.rglob("*") if file.is_file() and file.name != "task.json"}
        _atomic_json(self.task_dir / "task.json", self.task)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _verify_complete_task(self.task_dir, self.identity, synthetic=True)

    def test_history_epoch_count_and_numeric_fields_are_validated(self):
        history_path = self.task_dir / "completion-history.json"
        history_path.write_text(json.dumps([]) + "\n")
        with self.assertRaisesRegex(ValueError, "history length"):
            verify_task_artifacts(self.task_dir, self.task)
        history_path.write_text(json.dumps([{"epoch": 1, "objective": float("nan"),
            "reconstruction_mse": .9, "core_penalty": .1}]) + "\n")
        with self.assertRaisesRegex(ValueError, "objective must be finite"):
            verify_task_artifacts(self.task_dir, self.task)

    def test_wrong_completion_rank_is_rejected_after_checksum_recomputed(self):
        path = self.task_dir / "completion-state.npz"
        with np.load(path, allow_pickle=False) as state:
            arrays = {name: state[name].copy() for name in state.files}
        arrays["setting__rank_features"] = np.asarray(15, dtype=np.int64)
        _atomic_npz(path, **arrays)
        self.task["artifacts"] = {
            file.relative_to(self.task_dir).as_posix(): __import__("hashlib").sha256(file.read_bytes()).hexdigest()
            for file in self.task_dir.rglob("*") if file.is_file() and file.name != "task.json"}
        _atomic_json(self.task_dir / "task.json", self.task)
        with self.assertRaisesRegex(ValueError, "differs from the declared study"):
            _verify_complete_task(self.task_dir, self.identity, synthetic=True)


class HierarchicalReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = load_config(ROOT / "configs/completion-transformer.json", allow_existing_output=True)
        self.cfg["output_dir"] = str(Path(self.temp.name) / "out")
        initialize_output(self.cfg)
        for subject in range(1, 10):
            for seed in range(3):
                task_dir = Path(self.cfg["output_dir"]) / "tasks" / f"A{subject:02d}_seed_{seed}"
                task_dir.mkdir(parents=True)
                (task_dir / "task.json").write_text(json.dumps({
                    "historical_input_sha256": {"fixture": "0" * 64}}))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _task(subject, seed):
        rows = []
        transformer_gain = (subject - 5) * .01 + seed * .001
        for backbone in ARM_IDS:
            benefit = transformer_gain if backbone == "spatial_transformer" else transformer_gain - .002
            for condition in CONDITIONS:
                for repeat in range(condition["repeats"]):
                    for strategy in STRATEGY_IDS:
                        ba = .5
                        if condition["name"] != "full_22":
                            if strategy == "tucker":
                                ba += benefit
                            elif strategy == "covariance" and backbone == "spatial_transformer":
                                ba += benefit - .003
                        rows.append({"backbone": backbone, "strategy": strategy,
                            "condition": condition["name"], "repeat": repeat,
                            "balanced_accuracy": ba, "log_loss": 1.0,
                            "mask_sha256": "a" * 64, "artifact": "unused.npz"})
        return {"subject": subject, "seed": seed, "synthetic": False, "cells": rows}

    def _summarize_mocked(self, missing=None):
        def loader(path, identity, *, require_real):
            name = path.name.removeprefix("A")
            subject_text, seed_text = name.split("_seed_")
            subject, seed = int(subject_text), int(seed_text)
            if missing == (subject, seed):
                raise ValueError("deliberately missing task")
            return self._task(subject, seed)
        with patch("inm.completion_transformer.reporting._load_task", side_effect=loader), \
             patch("inm.completion_transformer.replay.verify_historical_task", return_value={}), \
             patch("inm.completion_transformer.study._historical_hashes", return_value={"fixture": "0" * 64}):
            return summarize(self.cfg)

    def test_complete_hierarchy_primary_secondaries_interaction_and_negative_effects(self):
        result = self._summarize_mocked()
        self.assertTrue(result["complete"])
        evidence = json.loads((Path(result["report_dir"]) / "evidence.json").read_text())
        rows = {row["contrast"]: row for row in evidence["contrasts"]}
        primary = rows["primary_transformer_tucker_minus_zero"]
        self.assertAlmostEqual(primary["mean_gain"], .001, places=12)
        self.assertEqual(primary["positive_participants"], 5)
        self.assertAlmostEqual(primary["participant_values"]["1"], -.039, places=12)
        self.assertAlmostEqual(rows["secondary_eegnet_tucker_minus_zero"]["mean_gain"], -.001, places=12)
        self.assertAlmostEqual(rows["secondary_transformer_tucker_minus_covariance"]["mean_gain"], .003, places=12)
        self.assertAlmostEqual(rows["interaction_transformer_minus_eegnet"]["mean_gain"], .002, places=12)
        self.assertEqual(primary["participants"], 9)
        self.assertEqual(len(primary["ci_95"]), 2)

    def test_one_missing_seed_retains_rows_but_suppresses_cohort_inference(self):
        result = self._summarize_mocked(missing=(1, 0))
        self.assertFalse(result["complete"])
        evidence = json.loads((Path(result["report_dir"]) / "evidence.json").read_text())
        self.assertEqual(evidence["valid_real_tasks"], 26)
        self.assertEqual(len(evidence["issues"]), 1)
        self.assertTrue(all(row["mean_gain"] is None and row["ci_95"] is None and
                            row["positive_participants"] is None for row in evidence["contrasts"]))
        with (Path(result["report_dir"]) / "cohort_conditions.csv").open() as stream:
            lines = stream.read().splitlines()
        self.assertTrue(lines[1].endswith(",,9"))
        self.assertGreater((Path(result["report_dir"]) / "per_repeat.csv").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
