"""Synthetic artifact boundary tests; no recordings or research fitting."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from inm.encoder_audit.artifacts import (ArtifactError, LABEL_NAMES, LEGACY_CHANNELS,
    _partition, load_task_sources)
from inm.encoder_audit.protocol import file_sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {"subjects": [1], "seeds": [0], "data_dir": str(self.root / "data"),
            "baseline_dir": str(self.root / "baseline"), "legacy_dir": str(self.root / "legacy"),
            "preprocessing": {"trial_samples": 1000, "window_samples": 250,
                "filter_low_hz": 2.0, "filter_high_hz": 30.0}}
        Path(self.cfg["data_dir"]).mkdir()
        recording = Path(self.cfg["data_dir"]) / "A01T.gdf"
        recording.write_bytes(b"synthetic source bytes")
        self.recording_hash = file_sha256(recording)
        self.ids = [f"A01T:cue:{i:03d}:sample:{i * 100}" for i in range(8)]
        self.labels = dict(zip(self.ids, [0, 1, 2, 3, 0, 1, 2, 3]))
        self._make_studies()

    def _make_studies(self):
        baseline = Path(self.cfg["baseline_dir"])
        legacy = Path(self.cfg["legacy_dir"])
        config_path = self.root / "baseline-config.json"
        write_json(config_path, {"fixture": True})
        base_identity = {"config": {"protocol": "within_session", "config_path": str(config_path)},
                         "config_sha256": file_sha256(config_path),
                         "source_sha256": {}}
        legacy_source = {"files_sha256": {}, "source_sha256": hashlib.sha256(b"{}").hexdigest(), "packages": {}}
        # Study IDs are independently derived exactly as recorded by the legacy format.
        from inm.encoder_audit.protocol import digest
        legacy_manifest = {"config": {"legacy": True}, "source": legacy_source,
            "study_id": digest({"config": {"legacy": True}, "source": legacy_source})}
        write_json(baseline / "study.json", {"identity": base_identity,
            "input_sha256": {str(Path(self.cfg["data_dir"]) / "A01T.gdf"): self.recording_hash}})
        write_json(legacy / "study.json", legacy_manifest)
        base_dir = baseline / "artifacts/A01_seed_0"
        legacy_dir = legacy / "artifacts/A01_seed_0"
        split = {"protocol": "stratified", "seed": 0, "split_id": "b" * 64,
            "fingerprint": "c" * 64, "identity": {"protocol": "stratified", "seed": 0, "fingerprint": "c" * 64},
            "train": [0, 1, 2, 3], "validation": [4, 5], "test": [6, 7],
            "sample_ids": {"train": self.ids[:4], "validation": self.ids[4:6], "test": self.ids[6:]}}
        legacy_dataset = {"sources": [{"name": "A01T.gdf", "sha256": self.recording_hash}],
                          "channel_names": list(LEGACY_CHANNELS)}
        base_dataset = {"channel_names": list(LEGACY_CHANNELS), "data_fingerprint": "d" * 64,
            "normalization_stats": {"mean": [0.0] * 22, "std": [1.0] * 22}}
        for directory in (base_dir, legacy_dir):
            write_json(directory / "split.json", split)
        write_json(base_dir / "dataset.json", base_dataset)
        write_json(legacy_dir / "dataset.json", legacy_dataset)
        legacy_cal = {"identity": {"study_id": legacy_manifest["study_id"], "subject": 1, "seed": 0,
                    "split_id": split["split_id"], "dataset_fingerprint": split["fingerprint"]},
            "cache_sha256": "", "feature_count": 32}
        tensor_payload = {"identity": legacy_cal["identity"], "features": torch.arange(8*22*4*32, dtype=torch.float32).reshape(8,22,4,32),
            "encoder": {"w": torch.ones(1)}, "tensor": {"U": torch.ones(22,4)},
            "raw_mean": torch.zeros(1,22,1), "raw_std": torch.ones(1,22,1),
            "feature_mean": torch.zeros(1,22,1,32), "feature_std": torch.ones(1,22,1,32),
            "encoder_selection": {"best_epoch": 2}}
        legacy_pt = legacy_dir / "calibration.pt"
        torch.save(tensor_payload, legacy_pt)
        legacy_cal["cache_sha256"] = file_sha256(legacy_pt)
        write_json(legacy_dir / "calibration.json", legacy_cal)
        for arm in ("eegnet_reference__full", "masked_eegnet__full", "masked_eegnet__mixed"):
            arm_dir = base_dir / "ARMS" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            history = [{"epoch": 1, "selection": {"balanced_accuracy": .4}},
                       {"epoch": 2, "selection": {"balanced_accuracy": .5}}]
            write_json(arm_dir / "history.json", history)
            checkpoint = {"format": "agfl-baseline-checkpoint-v1", "model_type": "tiny",
                          "constructor": {}, "state_dict": {"weight": torch.ones(1)}, "metadata": {}}
            checkpoint_path = arm_dir / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            result_identity = {"subject": 1, "seed": 0, "split_id": split["split_id"],
                "config_sha256": base_identity["config_sha256"],
                "source_sha256": base_identity["source_sha256"], "synthetic": False,
                "protocol": "within_session", "data_fingerprint": base_dataset["data_fingerprint"]}
            write_json(arm_dir / "result.json", {"subject": 1, "seed": 0, "status": "complete",
                "identity": result_identity,
                "selected_epoch": 2, "selection_policy": "validation_balanced_accuracy",
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "history_sha256": file_sha256(arm_dir / "history.json")})

    def load(self):
        with patch("inm.encoder_audit.artifacts._source_status", return_value={"verified.py": "match"}), \
             patch("inm.encoder_audit.artifacts._loader_labels", return_value=(self.labels, tuple(self.ids))):
            return load_task_sources(self.cfg, 1, 0)

    def test_train_validation_only_and_authoritative_ids(self):
        input_bytes = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        sources = self.load()
        for view in (sources.baseline, sources.legacy):
            self.assertEqual(view.train.sample_ids, tuple(self.ids[:4]))
            self.assertEqual(view.validation.sample_ids, tuple(self.ids[4:6]))
            self.assertEqual(view.train.labels.tolist(), [0, 1, 2, 3])
            self.assertEqual(view.validation.labels.tolist(), [0, 1])
            self.assertFalse(hasattr(view, "test"))
        self.assertEqual(sources.legacy.train_features.shape, (4, 22, 4, 32))
        self.assertEqual(sources.legacy.validation_features.shape, (2, 22, 4, 32))
        self.assertNotIn("test", repr(sources.legacy))
        self.assertEqual(sources.baseline.checkpoints["eegnet_reference__full"].selected_epoch, 2)
        self.assertEqual(sources.legacy.original_heads["pretraining_mha_head"], "unavailable_not_persisted")
        self.assertEqual(input_bytes, {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_checksum_rejected_before_deserialization(self):
        cache = Path(self.cfg["legacy_dir"]) / "artifacts/A01_seed_0/calibration.pt"
        before = cache.read_bytes()
        cache.write_bytes(before + b"corrupt")
        with patch("inm.encoder_audit.artifacts._torch_load", side_effect=AssertionError("deserialized")):
            with self.assertRaisesRegex(ArtifactError, "Checksum mismatch"):
                self.load()

    def test_swapped_identity_and_overlapping_split_fail(self):
        cal_path = Path(self.cfg["legacy_dir"]) / "artifacts/A01_seed_0/calibration.json"
        calibration = json.loads(cal_path.read_text())
        calibration["identity"]["seed"] = 1
        write_json(cal_path, calibration)
        with self.assertRaisesRegex(ArtifactError, "identity"):
            self.load()
        self._make_studies()
        split_path = Path(self.cfg["baseline_dir"]) / "artifacts/A01_seed_0/split.json"
        split = json.loads(split_path.read_text())
        split["validation"] = [3, 4]
        split["sample_ids"]["validation"] = [self.ids[3], self.ids[4]]
        write_json(split_path, split)
        with self.assertRaisesRegex(ArtifactError, "overlap"):
            self.load()

    def test_valid_but_unmatched_study_splits_are_preserved(self):
        split_path = Path(self.cfg["baseline_dir"]) / "artifacts/A01_seed_0/split.json"
        split = json.loads(split_path.read_text())
        split["validation"] = [5, 4]
        split["sample_ids"]["validation"] = [self.ids[5], self.ids[4]]
        write_json(split_path, split)
        sources = self.load()
        self.assertEqual(sources.baseline.validation.sample_ids, (self.ids[5], self.ids[4]))
        self.assertEqual(sources.legacy.validation.sample_ids, (self.ids[4], self.ids[5]))

    def test_poisoned_test_labels_or_features_do_not_change_views(self):
        first = self.load()
        legacy_pt = Path(self.cfg["legacy_dir"]) / "artifacts/A01_seed_0/calibration.pt"
        payload = torch.load(legacy_pt, map_location="cpu", weights_only=True)
        payload["features"][6:] = float("nan")
        torch.save(payload, legacy_pt)
        # Recompute only the recorded source checksum, as a source mutation would.
        cal_path = legacy_pt.with_name("calibration.json")
        cal = json.loads(cal_path.read_text())
        cal["cache_sha256"] = file_sha256(legacy_pt)
        write_json(cal_path, cal)
        self.labels[self.ids[6]] = 999
        self.labels[self.ids[7]] = -999
        second = self.load()
        np.testing.assert_array_equal(first.legacy.train_features, second.legacy.train_features)
        np.testing.assert_array_equal(first.legacy.validation_features, second.legacy.validation_features)
        np.testing.assert_array_equal(first.legacy.validation.labels, second.legacy.validation.labels)

    def test_partition_rejects_nonfinite_or_reordered_ids(self):
        split = {"protocol": "stratified", "seed": 0, "split_id": "s", "fingerprint": "f",
            "identity": {"protocol": "stratified", "seed": 0, "fingerprint": "f"},
            "train": [0, 1], "validation": [2], "test": [3],
            "sample_ids": {"train": [self.ids[1], self.ids[0]], "validation": [self.ids[2]], "test": [self.ids[3]]}}
        with self.assertRaisesRegex(ArtifactError, "IDs do not match"):
            _partition(split, {}, 1, 0, self.labels, tuple(self.ids[:4]))

    def test_nonfinite_train_features_are_rejected(self):
        legacy_pt = Path(self.cfg["legacy_dir"]) / "artifacts/A01_seed_0/calibration.pt"
        payload = torch.load(legacy_pt, map_location="cpu", weights_only=True)
        payload["features"][0, 0, 0, 0] = float("nan")
        torch.save(payload, legacy_pt)
        cal_path = legacy_pt.with_name("calibration.json")
        cal = json.loads(cal_path.read_text())
        cal["cache_sha256"] = file_sha256(legacy_pt)
        write_json(cal_path, cal)
        with self.assertRaisesRegex(ArtifactError, "Nonfinite"):
            self.load()


if __name__ == "__main__":
    unittest.main()
