"""Synthetic CPU alignment/replay acceptance; never opens a recording."""
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from inm.encoder_audit.alignment import AlignmentError, audit_alignment
from inm.encoder_audit.artifacts import (BaselineView, LegacyView, PartitionView,
    SourceMetadata, TaskSources, LABEL_NAMES, LEGACY_CHANNELS)
from inm.model import EEGWindowEncoder


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(12)
        self.raw = rng.normal(size=(8, 22, 1000)).astype(np.float32)
        self.ids = tuple(f"A01T:cue:{i:03}:sample:{i * 100}" for i in range(8))
        self.labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
        self.split = {"train": [0, 1, 2, 3], "validation": [4, 5], "test": [6, 7],
            "sample_ids": {"train": list(self.ids[:4]), "validation": list(self.ids[4:6]),
                "test": list(self.ids[6:])}}
        raw_train = self.raw[self.split["train"]].astype(np.float64)
        mean = raw_train.mean(axis=(0, 2)); std = np.maximum(raw_train.std(axis=(0, 2)), 1e-8)
        normalized = ((self.raw - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
        torch.manual_seed(4)
        self.encoder = EEGWindowEncoder().freeze()
        with torch.inference_mode():
            unscaled = self.encoder(torch.from_numpy(normalized)).numpy()
        train_features = unscaled[self.split["train"]]
        self.feature_mean = train_features.astype(np.float64).mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.feature_std = np.maximum(train_features.astype(np.float64).std(axis=(0, 2), keepdims=True), 1e-6).astype(np.float32)
        scaled = (unscaled - self.feature_mean) / self.feature_std
        self.cfg = {"data_dir": str(self.root / "data"), "preprocessing": {
            "trial_samples": 1000, "window_samples": 250, "filter_low_hz": 2.0,
            "filter_high_hz": 30.0}, "tolerances": {
                "features": {"rtol": 1e-4, "atol": 1e-5},
                "normalization": {"rtol": 1e-6, "atol": 1e-8}}}
        self.sources = self._sources(mean, std, scaled)
        self.bundle = SimpleNamespace(x=self.raw, y=self.labels, sample_ids=list(self.ids), metadata={
            "channel_names": list(LEGACY_CHANNELS), "label_names": list(LABEL_NAMES),
            "sampling_rate": 250.0, "preprocessing": {"offset_seconds": 0.0,
                "filter_scope": "window", "filter_window_samples": 250, "window": 1000,
                "lowcut": 2.0, "highcut": 30.0, "artifact_policy": "exclude"},
            "skipped": {"artifact": 0, "out_of_bounds": 0, "nonfinite": 0}})

    def _sources(self, mean, std, scaled):
        base_path = self.root / "base"; legacy_path = self.root / "legacy"
        for path in (base_path, legacy_path):
            path.mkdir(exist_ok=True)
            (path / "split.json").write_text(json.dumps(self.split))
            (path / "dataset.json").write_text(json.dumps({"source_exclusions": {
                "artifact": 0, "out_of_bounds": 0, "nonfinite": 0}}))
        base_part = lambda sl: PartitionView(self.ids[sl], self.labels[sl].copy())
        # Metadata task paths intentionally differ; only the saved split is read.
        baseline = BaselineView(base_part(slice(0, 4)), base_part(slice(4, 6)), mean, std, {})
        legacy = LegacyView(base_part(slice(0, 4)), base_part(slice(4, 6)),
            scaled[self.split["train"]].copy(), scaled[self.split["validation"]].copy(),
            mean.reshape(1, 22, 1), std.reshape(1, 22, 1), self.feature_mean,
            self.feature_std, self.encoder.state_dict(), {}, {}, {})
        metadata = lambda path: SourceMetadata("id", "src", "manifest", str(path), 1, 0,
            "split", "fingerprint", {}, {"inm/data.py": "match"})
        return TaskSources(1, 0, metadata(base_path), metadata(legacy_path), baseline, legacy)

    def run_audit(self, sources=None, bundle=None, ids=None, labels=None):
        with patch("inm.encoder_audit.alignment._load_source_bundle", return_value=bundle or self.bundle), \
             patch("inm.encoder_audit.artifacts._loader_labels", return_value=(
                 dict(zip(ids or self.ids, labels if labels is not None else self.labels)), tuple(ids or self.ids))):
            return audit_alignment(sources or self.sources, self.cfg)

    def test_cached_replay_batch_size_locality_and_hidden_nan(self):
        report = self.run_audit()
        self.assertEqual(report["status"], "complete")
        replay = report["legacy_feature_replay"]
        self.assertLessEqual(replay["train_cache_max_abs_error"], 1e-5)
        self.assertLessEqual(replay["validation_cache_max_abs_error"], 1e-5)
        self.assertEqual(replay["batchnorm_frozen"], True)
        self.assertEqual(replay["dropout_frozen"], True)
        self.assertEqual(replay["locality_max_abs_error"], 0.0)
        self.assertEqual(replay["hidden_nan_invariance_max_abs_error"], 0.0)

    def test_cue_label_shift_fails(self):
        shifted = self.labels.copy(); shifted[[0, 1]] = shifted[[1, 0]]
        bundle = SimpleNamespace(**{**vars(self.bundle), "y": shifted})
        with self.assertRaisesRegex(AlignmentError, "labels are shifted"):
            self.run_audit(bundle=bundle)

    def test_reordered_ids_fail(self):
        split_path = self.root / "legacy" / "split.json"
        split = json.loads(split_path.read_text()); split["sample_ids"]["train"][0], split["sample_ids"]["train"][1] = split["sample_ids"]["train"][1], split["sample_ids"]["train"][0]
        split_path.write_text(json.dumps(split))
        with self.assertRaisesRegex(AlignmentError, "reordered"):
            self.run_audit()

    def test_changed_normalization_fails(self):
        altered = self.sources.baseline.normalization_mean.copy(); altered[0] += 0.1
        sources = replace(self.sources, baseline=BaselineView(self.sources.baseline.train,
            self.sources.baseline.validation, altered, self.sources.baseline.normalization_std, {}))
        with self.assertRaisesRegex(AlignmentError, "baseline raw mean differs"):
            self.run_audit(sources=sources)

    def test_validation_contaminated_normalization_fails(self):
        all_mean = self.raw.astype(np.float64).mean(axis=(0, 2))
        all_std = np.maximum(self.raw.astype(np.float64).std(axis=(0, 2)), 1e-8)
        sources = replace(self.sources, baseline=BaselineView(self.sources.baseline.train,
            self.sources.baseline.validation, all_mean, all_std, {}))
        with self.assertRaisesRegex(AlignmentError, "baseline raw mean differs"):
            self.run_audit(sources=sources)

    def test_nonfinite_source_input_fails(self):
        bad = self.raw.copy(); bad[0, 0, 0] = np.nan
        bundle = SimpleNamespace(**{**vars(self.bundle), "x": bad})
        with self.assertRaisesRegex(AlignmentError, "nonfinite"):
            self.run_audit(bundle=bundle)

    def test_unmatched_cross_study_inputs_preserve_valid_within_study_results(self):
        val = PartitionView((self.ids[5], self.ids[4]), self.labels[[5, 4]])
        sources = replace(self.sources, baseline=BaselineView(self.sources.baseline.train, val,
            self.sources.baseline.normalization_mean, self.sources.baseline.normalization_std, {}))
        # A valid independently persisted baseline split can differ; arrange its stored membership.
        split = dict(self.split); split["validation"] = [5, 4]
        split["sample_ids"] = dict(self.split["sample_ids"], validation=[self.ids[5], self.ids[4]])
        (self.root / "base" / "split.json").write_text(json.dumps(split))
        report = self.run_audit(sources=sources)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["cross_study_pairing"]["status"], "unavailable_unmatched_inputs")

    def test_historical_source_mismatch_blocks_encoder_substitution(self):
        legacy_metadata = replace(self.sources.legacy_metadata,
            replay_source_status={"inm/model.py": "mismatch"})
        sources = replace(self.sources, legacy_metadata=legacy_metadata)
        with patch("inm.encoder_audit.alignment._legacy_replay", side_effect=AssertionError("must not replay")):
            report = self.run_audit(sources=sources)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["legacy_feature_replay"]["mismatches"]["legacy"], ["inm/model.py"])


if __name__ == "__main__":
    unittest.main()
