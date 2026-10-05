import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agfl.datasets.base import SignalDataset, canonical_json
from agfl.datasets.eeg import CHANNEL_NAMES
from inm.encoder_candidates.data import prepare_subject
from inm.encoder_candidates.protocol import load_config


class PairedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.cfg = load_config(Path(__file__).resolve().parents[2] / "configs/encoder-candidates.json")
        self.cfg["data_dir"] = str(root / "data")
        self.cfg["split_source_dir"] = str(root / "baseline")
        self.cfg["output_dir"] = str(root / "candidate-output")
        self.root = root
        self.bundle, self.ids, self.labels = self._make_bundle()
        self._persist(self.bundle, self.ids, self.labels)
        self.task = root / "new-task"

    def tearDown(self):
        self.temp.cleanup()

    def _make_bundle(self, x=None, y=None):
        rng = np.random.default_rng(510)
        labels = np.repeat(np.arange(4, dtype=np.int64), 8) if y is None else np.asarray(y, dtype=np.int64)
        if x is None:
            x = rng.normal(size=(32, 22, 1000)).astype(np.float32)
            x += labels[:, None, None] * np.float32(0.2)
        ids = [f"A01T:cue:{i:03d}:sample:{1000 + i}" for i in range(32)]
        source = {"name": "A01T.gdf", "sha256": "a" * 64, "bytes": 1234}
        loader_config = {"data_dir": self.cfg["data_dir"], "subjects": [1], "sessions": ["T"],
            "artifact_policy": "exclude", "filter_scope": "window", "window": 1000,
            "offset_seconds": 0.0, "filter_window_samples": 250, "lowcut": 2.0,
            "highcut": 30.0, "normalization": "none", "labels_dir": None}
        metadata = {"dataset": "eeg", "modality": "eeg", "num_classes": 4,
            "sampling_rate": 250.0, "channel_names": list(CHANNEL_NAMES),
            "label_names": ["left_hand", "right_hand", "feet", "tongue"],
            "preprocessing": loader_config, "sources": [source],
            "skipped": {"artifact": 0, "out_of_bounds": 0, "nonfinite": 0},
            "sample_sessions": ["T"] * 32, "sample_runs": ["A01T:run:0"] * 32,
            "protocol_version": "bci2a-inm-window-v1", "offline_zero_phase_filter": True}
        bundle = SignalDataset(x, labels, np.asarray(["A01"] * 32), ids, metadata)
        return bundle, ids, labels

    def _persist(self, bundle, ids, labels, *, split_override=None, dataset_override=None,
                 manifest_hash="a" * 64):
        directory = Path(self.cfg["split_source_dir"])
        task = directory / "artifacts/A01_seed_0"
        task.mkdir(parents=True, exist_ok=True)
        parts = {"train": [], "validation": [], "test": []}
        for start in range(0, 32, 8):
            members = list(range(start, start + 8))
            parts["train"].extend(members[:4])
            parts["validation"].extend(members[4:6])
            parts["test"].extend(members[6:])
        parts = {key: sorted(value) for key, value in parts.items()}
        identity = {"version": "agfl-splits-v2", "fingerprint": bundle.fingerprint,
                    "seed": 0, "config": {"protocol": "stratified", "train": 0.6,
                    "validation": 0.2, "test": 0.2}}
        split = {"split_id": hashlib.sha256(canonical_json(identity).encode()).hexdigest(),
            "fingerprint": bundle.fingerprint, "seed": 0, "protocol": "stratified",
            "identity": identity, "subject_independent": False, **parts,
            "sample_ids": {key: [ids[i] for i in values] for key, values in parts.items()},
            "class_counts": {key: np.bincount(labels[values], minlength=4).tolist()
                             for key, values in parts.items()}}
        split["partition_digest"] = hashlib.sha256(canonical_json(parts).encode()).hexdigest()
        for key, value in (split_override or {}).items():
            split[key] = value
        old_prep = {"channels": 22, "trial_samples": 1000, "windows": 4,
            "window_samples": 250, "sampling_rate": 250, "filter_low_hz": 2.0,
            "filter_high_hz": 30.0, "filter_scope": "window"}
        provenance = {"subject": 1, "seed": 0, "split_id": split["split_id"],
            "dataset_fingerprint": bundle.fingerprint,
            "sources": [{"name": "A01T.gdf", "sha256": manifest_hash, "bytes": 1234}],
            "source_exclusions": bundle.metadata["skipped"],
            "config": {"protocol": "within_session", "preprocessing": old_prep},
            "loader_config": bundle.metadata["preprocessing"]}
        normalization_stats = {"convention": "per_channel_train_trials_and_time_population_std",
            "mean": [0.0] * 22, "std": [1.0] * 22, "epsilon_floor": 1e-8}
        fingerprint_contents = {"provenance": provenance, "normalization": normalization_stats,
                                "sample_ids": ids}
        data_fingerprint = hashlib.sha256(canonical_json(fingerprint_contents).encode()).hexdigest()
        dataset = {"dataset_fingerprint": bundle.fingerprint, "data_fingerprint": data_fingerprint,
            "provenance": provenance, "sample_ids": ids, "channel_names": list(CHANNEL_NAMES),
            "source_exclusions": bundle.metadata["skipped"],
            "class_counts": split["class_counts"], "normalization_stats": normalization_stats}
        for key, value in (dataset_override or {}).items():
            dataset[key] = value
        (task / "split.json").write_text(json.dumps(split), encoding="utf-8")
        (task / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
        manifest = {"format": "agfl-baseline-study-v1", "synthetic": False,
            "identity": {"config": {"protocol": "within_session"}},
            "input_sha256": {str(Path(self.cfg["data_dir"]) / "A01T.gdf"): manifest_hash}}
        (directory / "study.json").write_text(json.dumps(manifest), encoding="utf-8")
        return split, dataset, manifest

    def _load(self, bundle=None, subject=1, seed=0):
        return prepare_subject(self.cfg, subject, seed, self.task,
            loader=lambda _: self.bundle if bundle is None else bundle)

    def test_canonical_shape_order_and_train_reference_normalization(self):
        result = self._load()
        indices = result.provenance["partition_indices"]["train"]
        x = self.bundle.x[indices].astype(np.float64)
        mean = x.mean(axis=(0, 2))
        std = np.maximum(x.std(axis=(0, 2), ddof=0), 1e-8)
        self.assertEqual(result.train.raw.shape, (16, 22, 4, 250))
        self.assertEqual(result.validation.raw.shape, (8, 22, 4, 250))
        self.assertEqual(result.train.sample_ids, tuple(self.ids[i] for i in indices))
        expected = ((self.bundle.x[indices].astype(np.float64) - mean[None, :, None]) /
                    std[None, :, None]).astype(np.float32).reshape(16, 22, 4, 250)
        np.testing.assert_allclose(result.train.raw, expected, rtol=0, atol=0)
        np.testing.assert_allclose(result.normalization["mean"], mean, rtol=0, atol=0)
        np.testing.assert_allclose(result.normalization["std"], std, rtol=0, atol=0)
        self.assertEqual(tuple(result.provenance["channel_names"]), tuple(CHANNEL_NAMES))
        self.assertFalse(hasattr(result, "test"))
        with self.assertRaises(ValueError):
            result.train.raw[0, 0, 0, 0] = 1

    def test_test_partition_poison_is_not_returned_or_used(self):
        first = self._load()
        persisted = json.loads((Path(self.cfg["split_source_dir"]) /
                                "artifacts/A01_seed_0/split.json").read_text())
        test_idx = persisted["test"]
        poisoned_x = self.bundle.x.copy()
        poisoned_y = self.bundle.y.copy()
        poisoned_x[test_idx] = 1e6
        poisoned_y[test_idx[0]], poisoned_y[test_idx[2]] = poisoned_y[test_idx[2]], poisoned_y[test_idx[0]]

        class LoadedFixtureView:
            def __init__(self, original, x, y):
                self.x, self.y = x, y
                self.sample_ids, self.metadata = original.sample_ids, original.metadata
                self._identity = original.fingerprint

            @property
            def fingerprint(self):
                return self._identity

            def __len__(self):
                return len(self.y)

        second = self._load(LoadedFixtureView(self.bundle, poisoned_x, poisoned_y))
        np.testing.assert_array_equal(first.train.raw, second.train.raw)
        np.testing.assert_array_equal(first.validation.raw, second.validation.raw)
        np.testing.assert_array_equal(first.train.labels, second.train.labels)
        np.testing.assert_array_equal(first.validation.labels, second.validation.labels)
        self.assertEqual(first.normalization, second.normalization)
        self.assertFalse(any(np.shares_memory(part.raw, poisoned_x) for part in
                             (second.train, second.validation)))

    def test_validation_signal_poison_does_not_change_training_normalization(self):
        first = self._load()
        val_idx = first.provenance["partition_indices"]["validation"]
        changed = self.bundle.x.copy()
        changed[val_idx] = changed[val_idx] * 1000 + 77
        variant, ids, labels = self._make_bundle(changed, self.bundle.y.copy())
        # Rebuild metadata to match this independently loaded fixture identity.
        original_task = Path(self.cfg["split_source_dir"]) / "artifacts/A01_seed_0"
        for path in (original_task / "split.json", original_task / "dataset.json"):
            path.unlink()
        self._persist(variant, ids, labels)
        second = self._load(variant)
        self.assertEqual(first.normalization, second.normalization)
        np.testing.assert_array_equal(first.train.raw, second.train.raw)
        self.assertFalse(np.array_equal(first.validation.raw, second.validation.raw))

    def test_overlap_missing_class_wrong_seed_and_reordered_membership_fail(self):
        split, _, _ = self._persist(self.bundle, self.ids, self.labels)
        overlap = list(split["validation"])
        overlap[0] = split["train"][0]
        overlap.sort()
        self._persist(self.bundle, self.ids, self.labels,
                      split_override={"validation": overlap,
                          "sample_ids": {**split["sample_ids"],
                              "validation": [self.ids[i] for i in overlap]}})
        with self.assertRaisesRegex(ValueError, "overlap"):
            self._load()
        self._persist(self.bundle, self.ids, self.labels)
        with self.assertRaisesRegex(ValueError, "seed"):
            self._load(seed=1)
        order = list(split["train"])
        order.reverse()
        self._persist(self.bundle, self.ids, self.labels,
                      split_override={"train": order})
        with self.assertRaises(ValueError):
            self._load()

    def test_mismatched_recording_hash_and_missing_classes_fail(self):
        self._persist(self.bundle, self.ids, self.labels, manifest_hash="c" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self._load()
        missing = self.bundle.y.copy()
        missing[:4] = 1
        changed, ids, labels = self._make_bundle(self.bundle.x.copy(), missing)
        self._persist(changed, ids, labels)
        with self.assertRaisesRegex(ValueError, "class"):
            self._load(changed)

    def test_input_metadata_bytes_are_unchanged_and_provenance_is_new_task_only(self):
        source = Path(self.cfg["split_source_dir"])
        before = {p: p.read_bytes() for p in (source / "study.json",
            source / "artifacts/A01_seed_0/split.json",
            source / "artifacts/A01_seed_0/dataset.json")}
        result = self._load()
        self.assertTrue((self.task / "data_provenance.json").is_file())
        self.assertEqual(result.provenance["original_metadata_sha256"].keys(),
                         {"split.json", "dataset.json", "study.json"})
        self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == "__main__":
    unittest.main()
