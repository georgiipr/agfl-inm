from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inm.completion_transformer import replay
from inm.completion_transformer.protocol import ARM_IDS, file_sha256

ROOT = Path(__file__).resolve().parents[2]
HIST_CANDIDATE = ROOT / "results/encoder-candidates-reproducible-v1"
HIST_SPLITS = ROOT / "results/baselines-reproducible-v1"
TASK_REL = "tasks/A01_seed_0/task.json"
REUSED = sorted(replay.REUSED_SOURCE_KEYS)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class HistoricalReplayFixture(unittest.TestCase):
    """A compact copy of the historical artifact schema with real saved IDs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.candidate = self.root / "candidate"
        self.splits = self.root / "splits"
        self.task_path = self.candidate / TASK_REL
        self.split_dir = self.splits / "artifacts/A01_seed_0"
        self.candidate.mkdir()
        self.splits.mkdir()
        (self.root / "configs").mkdir()
        shutil.copy2(ROOT / "configs/encoder-candidates-reproducible.json",
                     self.root / "configs/encoder-candidates-reproducible.json")

        source_map = {}
        for relative in REUSED:
            src = ROOT / relative
            dst = self.root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            source_map[relative] = file_sha256(src)

        original_evidence = json.loads((HIST_CANDIDATE / "report/evidence.json").read_text())
        task_record = next(item for item in original_evidence["tasks"] if item["path"] == TASK_REL)
        task_dir = HIST_CANDIDATE / Path(TASK_REL).parent
        self.task_path.parent.mkdir(parents=True)
        shutil.copy2(task_dir / "task.json", self.task_path)
        shutil.copy2(task_dir / "data_provenance.json", self.task_path.parent / "data_provenance.json")
        task = json.loads(self.task_path.read_text())
        task["source_files_sha256"] = source_map
        _write_json(self.task_path, task)

        fits = []
        for arm in ARM_IDS:
            old_dir = task_dir / "ARMS" / arm
            new_dir = self.task_path.parent / "ARMS" / arm
            new_dir.mkdir(parents=True)
            for artifact in ("checkpoint.pt", "result.json", "predictions.npz", "history.json"):
                shutil.copy2(old_dir / artifact, new_dir / artifact)
            fits.append({"arm": arm, "path": f"ARMS/{arm}/result.json",
                         "sha256": file_sha256(new_dir / "result.json")})
        task["fits"] = fits
        _write_json(self.task_path, task)

        (self.candidate / "report").mkdir()
        evidence = copy.deepcopy(original_evidence)
        evidence["source_files_sha256"] = source_map
        evidence["tasks"] = [
            ({**row, "sha256": file_sha256(self.task_path)} if row["path"] == TASK_REL else row)
            for row in evidence["tasks"]
        ]
        _write_json(self.candidate / "report/evidence.json", evidence)

        self.split_dir.mkdir(parents=True)
        for name in ("split.json", "dataset.json"):
            shutil.copy2(HIST_SPLITS / "artifacts/A01_seed_0" / name, self.split_dir / name)
        shutil.copy2(HIST_SPLITS / "study.json", self.splits / "study.json")
        self.cfg = {"candidate_source_dir": str(self.candidate), "split_source_dir": str(self.splits),
                    "data_dir": json.loads((self.split_dir / "dataset.json").read_text())
                        ["provenance"]["loader_config"]["data_dir"],
                    "output_dir": str(self.root / "new-output"),
                    "subjects": [1], "seeds": [0]}
        self.root_patch = patch.object(replay, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def refresh_task_manifest(self):
        evidence_path = self.candidate / "report/evidence.json"
        evidence = json.loads(evidence_path.read_text())
        evidence["tasks"][0]["sha256"] = file_sha256(self.task_path)
        _write_json(evidence_path, evidence)

    def test_historical_schema_verifies_and_restores_both_backbones(self):
        record = replay.verify_historical_task(self.cfg, 1, 0)
        models = replay.restore_backbones(record)
        self.assertEqual(set(models), set(ARM_IDS))
        self.assertTrue(all(not model.training for model in models.values()))

    def test_changed_checkpoint_rejected_by_saved_artifact_hash(self):
        path = self.task_path.parent / "ARMS/spatial_eegnet/checkpoint.pt"
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "checkpoint.pt checksum"):
            replay.verify_historical_task(self.cfg, 1, 0)

    def test_source_drift_rejected_even_when_task_checksums_are_recomputed(self):
        source = self.root / REUSED[0]
        source.write_bytes(source.read_bytes() + b" drift")
        with self.assertRaisesRegex(ValueError, "source checksum"):
            replay.verify_historical_task(self.cfg, 1, 0)

    def test_regenerated_task_hash_cannot_hide_split_id_mismatch(self):
        task = json.loads(self.task_path.read_text())
        task["split_id"] = "0" * 64
        provenance_path = self.task_path.parent / "data_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["split_id"] = task["split_id"]
        _write_json(provenance_path, provenance)
        _write_json(self.task_path, task)
        self.refresh_task_manifest()
        with self.assertRaisesRegex(ValueError, "baseline split ID"):
            replay.verify_historical_task(self.cfg, 1, 0)

    def test_regenerated_task_hash_cannot_hide_normalization_mismatch(self):
        task = json.loads(self.task_path.read_text())
        task["normalization"]["mean"][0] += 0.25
        provenance_path = self.task_path.parent / "data_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["normalization"] = task["normalization"]
        _write_json(provenance_path, provenance)
        _write_json(self.task_path, task)
        self.refresh_task_manifest()
        with self.assertRaisesRegex(ValueError, "baseline normalization mean"):
            replay.verify_historical_task(self.cfg, 1, 0)

    def test_regenerated_task_hash_cannot_hide_partition_id_mismatch(self):
        task = json.loads(self.task_path.read_text())
        task["partition_sample_ids"]["validation"][0] = "A01T:forged:trial"
        provenance_path = self.task_path.parent / "data_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["partition_sample_ids"] = task["partition_sample_ids"]
        _write_json(provenance_path, provenance)
        _write_json(self.task_path, task)
        self.refresh_task_manifest()
        with self.assertRaisesRegex(ValueError, "ordered train/validation IDs"):
            replay.verify_historical_task(self.cfg, 1, 0)

    def test_prepare_refuses_staging_outside_new_output_or_occupied_stage(self):
        outside = self.root / "outside"
        with self.assertRaisesRegex(ValueError, "strictly inside the new output"):
            replay.prepare_task(self.cfg, 1, 0, outside)
        stage = Path(self.cfg["output_dir"]) / "A01_seed_0"
        stage.mkdir(parents=True)
        (stage / "sentinel").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already contains files"):
            replay.prepare_task(self.cfg, 1, 0, stage)
        self.assertEqual((stage / "sentinel").read_text(), "keep")


class NumericalReplayTests(unittest.TestCase):
    def test_restored_predictions_match_saved_values_and_reject_mismatch(self):
        import torch
        from inm.encoder_candidates.models import build_model

        rng = np.random.default_rng(44)
        raw = rng.normal(size=(2, 22, 4, 250)).astype(np.float32)
        labels = np.asarray([0, 2], dtype=np.int64)
        ids = ["synthetic-val-0", "synthetic-val-1"]
        normalization = {"convention": "per_channel_train_trials_and_time_population_std",
                         "mean": [0.0] * 22, "std": [1.0] * 22, "epsilon_floor": 1e-8,
                         "axes": ["training_trials", "time"]}
        with tempfile.TemporaryDirectory() as tmp:
            fits = {}
            for arm in ARM_IDS:
                model = build_model(arm, 0).eval()
                with torch.inference_mode():
                    probabilities = torch.softmax(model(torch.from_numpy(raw),
                                                        torch.ones((2, 22, 4), dtype=torch.bool)), -1).numpy()
                directory = Path(tmp) / arm
                directory.mkdir()
                compatibility = {"arm": arm, "config_sha256": "c" * 64, "constructor": model.constructor_settings(),
                    "data_id": "d" * 64, "execution_device": "cpu", "original_metadata_sha256": {},
                    "recording_sha256": {}, "rng_policy": {}, "seed": 0, "split_id": "s" * 64,
                    "study_id": "study", "subject": 1, "synthetic": False}
                checkpoint = {"format": "agfl-encoder-candidate-classifier-v1",
                    "constructor": model.constructor_settings(), "state_dict": model.state_dict(),
                    "metadata": compatibility, "normalization": normalization, "selected_epoch": 1,
                    "channel_order": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
                    "class_order": replay.CLASS_ORDER}
                torch.save(checkpoint, directory / "checkpoint.pt")
                np.savez(directory / "predictions.npz", validation_sample_ids=np.asarray(ids),
                         validation_labels=labels, validation_probabilities=probabilities)
                fit_result = {"constructor": model.constructor_settings(), "compatibility": compatibility,
                              "selected_epoch": 1}
                fits[arm] = {"directory": directory, "result": fit_result}

            record = {"subject": 1, "seed": 0, "fits": fits,
                      "task": {"normalization": normalization, "data_id": "d" * 64, "split_id": "s" * 64,
                               "channel_order": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]},
                      "evidence": {"study_id": "study"}}
            prepared = SimpleNamespace(validation=SimpleNamespace(raw=raw, labels=labels, sample_ids=ids))
            replayed = replay.replay_full_input(record, prepared, batch_size=1)
            self.assertEqual(set(replayed), set(ARM_IDS))

            bad_path = fits["spatial_transformer"]["directory"] / "predictions.npz"
            np.savez(bad_path, validation_sample_ids=np.asarray(ids), validation_labels=labels,
                     validation_probabilities=np.zeros((2, 4), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "do not replay"):
                replay.replay_full_input(record, prepared)


if __name__ == "__main__":
    unittest.main()
