"""CPU-only checks for baseline data preparation and leakage boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from agfl.datasets.base import SignalDataset
from inm.baselines.data import make_synthetic_signal_dataset, prepare_subject
from inm.baselines.protocol import load_config


ROOT = Path(__file__).resolve().parents[2]


class BaselineDataTests(unittest.TestCase):
    def _config(self, protocol="within_session", external_labels_dir=None):
        cfg = load_config(ROOT / "configs" / ("baselines-cross-session.json" if protocol == "cross_session"
                                                else "baselines.json"))
        cfg["external_labels_dir"] = external_labels_dir
        return cfg

    def test_within_session_uses_t_and_persists_normalization_and_provenance(self):
        cfg = self._config()
        dataset = make_synthetic_signal_dataset(4, samples_per_class=20)
        dataset.metadata["sources"] = [{"name": "synthetic.bin", "sha256": "abc", "bytes": 3}]
        dataset.metadata["skipped"] = {"artifact": 2, "out_of_bounds": 1}
        received = {}

        def loader(loader_cfg):
            received.update(loader_cfg)
            return dataset

        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_subject(cfg, 1, 2, tmp, loader=loader)
            persisted = json.loads((Path(tmp) / "dataset.json").read_text())
            split = json.loads((Path(tmp) / "split.json").read_text())
            self.assertEqual(received["sessions"], ["T"])
            self.assertEqual(received["normalization"], "none")
            self.assertEqual(received["filter_window_samples"], 250)
            self.assertEqual(prepared.filtered_signals.shape, (80, 22, 1000))
            self.assertEqual(prepared.signals.shape, prepared.filtered_signals.shape)
            self.assertEqual(set(prepared.split_indices), {"train", "validation", "test"})
            for name, indices in prepared.split_indices.items():
                self.assertEqual(set(prepared.metadata["sample_sessions"][i] for i in indices), {"T"})
                self.assertTrue(np.all(np.asarray(prepared.metadata["class_counts"][name]) > 0))
            self.assertEqual(split["split_id"], prepared.metadata["split_id"])
            self.assertEqual(persisted["provenance"]["config"]["config_sha256"],
                             prepared.metadata["provenance"]["config"]["config_sha256"])
            self.assertEqual(persisted["provenance"]["sources"], dataset.metadata["sources"])
            self.assertEqual(persisted["source_exclusions"], dataset.metadata["skipped"])
            train = prepared.signals[prepared.split_indices["train"]]
            np.testing.assert_allclose(train.mean(axis=(0, 2)), np.zeros(22), atol=2e-7)
            np.testing.assert_allclose(train.std(axis=(0, 2)), np.ones(22), atol=2e-6)
            self.assertEqual(len(prepared.channel_names), 22)

    def test_validation_and_test_changes_do_not_change_training_statistics(self):
        cfg = self._config()
        original = make_synthetic_signal_dataset(5, samples_per_class=20)
        altered_x = original.x.copy()
        with tempfile.TemporaryDirectory() as tmp:
            first = prepare_subject(cfg, 1, 7, Path(tmp) / "first", loader=lambda _: original)
            held_out = np.concatenate([first.split_indices["validation"], first.split_indices["test"]])
            altered_x[held_out] += np.float32(10000)
            altered = SignalDataset(altered_x, original.y.copy(), original.groups.copy(),
                original.sample_ids.copy(), json.loads(json.dumps(original.metadata)))
            second = prepare_subject(cfg, 1, 7, Path(tmp) / "second", loader=lambda _: altered)
            np.testing.assert_array_equal(first.normalization_stats["mean"], second.normalization_stats["mean"])
            np.testing.assert_array_equal(first.normalization_stats["std"], second.normalization_stats["std"])
            self.assertFalse(np.array_equal(first.filtered_signals, second.filtered_signals))

    def test_cross_session_requires_explicit_labels_and_keeps_e_exclusively_test(self):
        cfg = self._config("cross_session")
        dataset = make_synthetic_signal_dataset(9, samples_per_class=20, sessions=("T", "E"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires external_labels_dir"):
                prepare_subject(cfg, 1, 0, tmp, loader=lambda _: self.fail("loader must not run"))
            cfg["external_labels_dir"] = str(Path(tmp) / "official-labels")
            received = {}

            def loader(loader_cfg):
                received.update(loader_cfg)
                return dataset

            prepared = prepare_subject(cfg, 1, 0, Path(tmp) / "prepared", loader=loader)
            self.assertEqual(received["sessions"], ["T", "E"])
            sessions = np.asarray(prepared.metadata["sample_sessions"])
            self.assertTrue(np.all(sessions[prepared.split_indices["train"]] == "T"))
            self.assertTrue(np.all(sessions[prepared.split_indices["validation"]] == "T"))
            self.assertTrue(np.all(sessions[prepared.split_indices["test"]] == "E"))
            self.assertEqual(len(prepared.split_indices["test"]), 80)

    def test_persisted_split_is_deterministic(self):
        cfg = self._config()
        dataset = make_synthetic_signal_dataset(11, samples_per_class=20)
        with tempfile.TemporaryDirectory() as tmp:
            a = prepare_subject(cfg, 1, 12, Path(tmp) / "a", loader=lambda _: dataset)
            b = prepare_subject(cfg, 1, 12, Path(tmp) / "b", loader=lambda _: dataset)
            self.assertEqual(a.metadata["split_id"], b.metadata["split_id"])
            self.assertEqual(a.data_fingerprint, b.data_fingerprint)
            for partition in ("train", "validation", "test"):
                np.testing.assert_array_equal(a.split_indices[partition], b.split_indices[partition])

    def test_rejects_sampling_metadata_mismatch(self):
        cfg = self._config()
        dataset = make_synthetic_signal_dataset(13, samples_per_class=20)
        dataset.metadata["sampling_rate"] = 200.0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "sampling rate"):
                prepare_subject(cfg, 1, 0, tmp, loader=lambda _: dataset)


if __name__ == "__main__":
    unittest.main()
