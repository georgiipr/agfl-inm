"""Synthetic task orchestration, provenance, resume, and failure checks."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from inm.encoder_candidates import study
from inm.encoder_candidates.protocol import load_config


class CandidateStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source_config = Path(__file__).resolve().parents[2] / "configs/encoder-candidates.json"
        self.cfg = load_config(source_config)
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg.update({"synthetic": True, "subjects": [1], "seeds": [0],
            "data_dir": str(self.root / "data"),
            "split_source_dir": str(self.root / "splits"),
            "output_dir": str(self.root / "output")})
        self.cfg["training"].update({"maximum_epochs": 1, "minimum_epochs": 1,
            "patience": 0, "batch_size": 8, "warmup_epochs": 0})
        self.prepared = study._synthetic_prepared(per_class=2)
        self.prepared.provenance.update({"recording_sha256": {"A01T.gdf": "a" * 64},
            "original_metadata_sha256": {"split.json": "b" * 64, "dataset.json": "c" * 64,
                                          "study.json": "d" * 64}})

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, **patches):
        with patch.object(study, "prepare_subject", return_value=self.prepared):
            return study.run_task(self.cfg, 0, device="cpu", **patches)

    def test_five_arms_artifacts_resume_checks_and_test_field_absence(self):
        result = self._run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual([row["arm"] for row in result["fits"]], [
            "local_control", "local_power", "spatial_eegnet", "spatial_filterbank",
            "spatial_transformer"])
        self.assertTrue(all(result["checks"].values()))
        root = Path(self.cfg["output_dir"]) / "tasks/A01_seed_0"
        self.assertTrue((root / "data_provenance.json").is_file())
        for record in result["fits"]:
            fit_result = json.loads((root / record["path"]).read_text())
            self.assertEqual(len(fit_result["validation"]), 21)
            self.assertIn("predictions.npz", fit_result["artifact_sha256"])
            self.assertTrue((root / "ARMS" / record["arm"] / "checkpoint.pt").is_file())
            if record["arm"].startswith("local_"):
                self.assertTrue(fit_result["probe_converged"])
                self.assertTrue(fit_result["probe_shuffled_converged"])
        manifest_bytes = (root / "task.json").read_bytes()
        resumed = self._run()
        self.assertEqual(result, resumed)
        self.assertEqual(manifest_bytes, (root / "task.json").read_bytes())
        manifest = json.loads(manifest_bytes)
        self.assertNotIn("test", manifest)
        for record in manifest["fits"]:
            fit_data = json.loads((root / record["path"]).read_text())
            self.assertNotIn("test", fit_data)

    def test_config_source_package_and_artifact_tampering_reject_resume(self):
        result = self._run()
        task_path = Path(self.cfg["output_dir"]) / "tasks/A01_seed_0/task.json"
        with patch.object(study, "study_identity", wraps=study.study_identity) as identify, \
             patch.object(study, "prepare_subject", return_value=self.prepared):
            changed_identity = study.study_identity(self.cfg) | {"packages": {"torch": "changed"}}
            identify.return_value = changed_identity
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                study.run_task(self.cfg, 0, loader=None)

        with patch.object(study, "study_identity", wraps=study.study_identity) as identify, \
             patch.object(study, "prepare_subject", return_value=self.prepared):
            changed_identity = study.study_identity(self.cfg) | {
                "source_files_sha256": {"inm/encoder_candidates/study.py": "e" * 64},
                "study_id": "f" * 64}
            identify.return_value = changed_identity
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                study.run_task(self.cfg, 0, loader=None)

        alternate = self.root / "changed-config.json"
        original_config = Path(self.cfg["config_path"])
        alternate.write_bytes(original_config.read_bytes() + b" ")
        prior_config = self.cfg["config_path"]
        self.cfg["config_path"] = str(alternate)
        with patch.object(study, "prepare_subject", return_value=self.prepared):
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                study.run_task(self.cfg, 0, loader=None)
        self.cfg["config_path"] = prior_config

        fit_path = Path(self.cfg["output_dir"]) / "tasks/A01_seed_0" / result["fits"][0]["path"]
        payload = json.loads(fit_path.read_text())
        payload["artifact_sha256"]["history.json"] = "0" * 64
        fit_path.write_text(json.dumps(payload))
        with patch.object(study, "prepare_subject", return_value=self.prepared):
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                study.run_task(self.cfg, 0, loader=None)

        task_path.unlink()
        # A nonempty output without a task identity is never treated as fresh.
        with patch.object(study, "prepare_subject", return_value=self.prepared):
            with self.assertRaisesRegex(FileExistsError, "partial task artifacts"):
                study.run_task(self.cfg, 0, loader=None)

    def test_failed_arm_is_journaled_and_requires_fresh_output(self):
        def fail_on_first_arm(*args, **kwargs):
            raise RuntimeError("fixture diagnostic failure")

        with patch.object(study, "prepare_subject", return_value=self.prepared), \
             patch.object(study, "evaluate_selected", side_effect=fail_on_first_arm):
            with self.assertRaisesRegex(RuntimeError, "fixture diagnostic failure"):
                study.run_task(self.cfg, 0)
        root = Path(self.cfg["output_dir"]) / "tasks/A01_seed_0"
        journal = json.loads((root / "task.json").read_text())
        self.assertEqual(journal["status"], "failed")
        self.assertEqual(journal["failure"]["stage"], "evaluate:local_control")
        self.assertTrue((root / "ARMS/local_control/checkpoint.pt").is_file())
        with patch.object(study, "prepare_subject", return_value=self.prepared):
            with self.assertRaisesRegex(ValueError, "restart under a fresh output"):
                study.run_task(self.cfg, 0)

    def test_input_failure_is_journaled_and_output_symlink_cannot_escape(self):
        with patch.object(study, "prepare_subject", side_effect=FileNotFoundError("fixture input missing")):
            with self.assertRaisesRegex(FileNotFoundError, "fixture input missing"):
                study.run_task(self.cfg, 0)
        task_root = Path(self.cfg["output_dir"]) / "tasks/A01_seed_0"
        manifest = json.loads((task_root / "task.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"]["stage"], "prepare_subject")

        output = self.root / "path-check-output"
        output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (output / "tasks").symlink_to(outside, target_is_directory=True)
        cfg = copy.deepcopy(self.cfg)
        cfg["output_dir"] = str(output)
        with self.assertRaisesRegex(ValueError, "escapes declared output root"):
            study._task_root(cfg, 1, 0)

    def test_smoke_trains_all_arms_and_rejects_overlapping_paths(self):
        smoke_dir = self.root / "smoke"
        result = study.run_smoke(self.cfg, smoke_dir)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["report"]["complete"])
        self.assertEqual(result["report"]["status"], "partial")
        evidence = json.loads((smoke_dir / "report/evidence.json").read_text())
        self.assertEqual(evidence["status"], "partial")
        self.assertEqual(evidence["tasks"], [])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["fits"]), 5)
        for arm in ("local_control", "local_power", "spatial_eegnet",
                    "spatial_filterbank", "spatial_transformer"):
            directory = smoke_dir / "tasks/A01_seed_0/ARMS" / arm
            self.assertTrue((directory / "checkpoint.pt").is_file())
            self.assertTrue((directory / "predictions.npz").is_file())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            study.run_smoke(self.cfg, self.cfg["output_dir"])


if __name__ == "__main__":
    unittest.main()
