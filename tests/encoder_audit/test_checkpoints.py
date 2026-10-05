"""Tiny synthetic clean-checkpoint replay tests; no recordings or fitting."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from inm.baselines.eegnet import EEGNetClassifier
from inm.encoder_audit.artifacts import (BaselineCheckpoint, BaselineView, LegacyView,
    PartitionView, SourceMetadata, TaskSources, LABEL_NAMES, LEGACY_CHANNELS)
from inm.encoder_audit.checkpoints import CheckpointAuditError, audit_checkpoints
from inm.training import metrics


class CheckpointAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(12, 22, 1000)).astype(np.float32)
        self.y = np.tile(np.arange(4), 3)
        self.ids = tuple(f"A01T:cue:{i:03}:sample:{i * 100}" for i in range(12))
        self.mean, self.std = np.zeros(22), np.ones(22)
        parts = {"train": np.arange(0, 4), "validation": np.arange(4, 8)}
        self.cfg = {"partitions": ["train", "validation"], "checkpoint_batch_size": 2,
            "tolerances": {"probabilities": {"rtol": 1e-4, "atol": 1e-5}}}
        self.aligned = {"within_study": {"baseline": {"status": "verified"}}}
        self.sources = self._make_sources(parts)
        self.bundle = SimpleNamespace(x=self.x.copy(), y=self.y.copy(), sample_ids=list(self.ids), metadata={
            "channel_names": list(LEGACY_CHANNELS), "label_names": list(LABEL_NAMES)})

    def _make_sources(self, parts, *, bad_epoch=None, bad_class_order=False):
        base_dir = self.root / "baseline_task"
        artifact_hashes = {}
        checkpoints = {}
        for arm in ("eegnet_reference__full", "masked_eegnet__full", "masked_eegnet__mixed"):
            arm_dir = base_dir / "ARMS" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = arm_dir / "checkpoint.pt"
            checkpoint_file.write_bytes((arm + "-checkpoint").encode())
            history_file = arm_dir / "history.json"
            history_file.write_bytes((arm + "-history").encode())
            artifact_hashes[f"{arm}/checkpoint.pt"] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
            artifact_hashes[f"{arm}/history.json"] = hashlib.sha256(history_file.read_bytes()).hexdigest()
            model = EEGNetClassifier(mask_conditioned=arm != "eegnet_reference__full")
            model.eval()
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            selected_epoch = 2
            metadata = {"selected_epoch": selected_epoch, "subject": 1, "seed": 0, "arm": arm,
                "class_order": list(reversed(LABEL_NAMES)) if bad_class_order else list(LABEL_NAMES),
                "channel_order": list(LEGACY_CHANNELS),
                "normalization_stats": {"mean": self.mean.tolist(), "std": self.std.tolist()}}
            history_row = {"epoch": selected_epoch, "masked_train_loss": 1.4,
                "masked_train_accuracy": 0.31}
            for part, indices in parts.items():
                raw = self.x[indices].reshape(-1, 22, 4, 250)
                with torch.no_grad():
                    probs = model(torch.from_numpy(raw)).softmax(-1).numpy()
                clean = metrics(self.y[indices], probs)
                clean["log_loss"] = float(-np.log(np.clip(probs[np.arange(len(indices)), self.y[indices]], 1e-12, 1)).mean())
                history_row[f"clean_{part}_metrics"] = clean
            checkpoints[arm] = BaselineCheckpoint(arm, bad_epoch or selected_epoch,
                "full", "inm.baselines.eegnet.EEGNetClassifier", model.constructor_settings(),
                state, metadata, (history_row,))
        partition = lambda name: PartitionView(tuple(self.ids[i] for i in parts[name]), self.y[parts[name]].copy())
        baseline = BaselineView(partition("train"), partition("validation"), self.mean.copy(), self.std.copy(), checkpoints)
        legacy_partition = PartitionView((), np.asarray([], dtype=np.int64))
        legacy = LegacyView(legacy_partition, legacy_partition, np.empty((0,)), np.empty((0,)),
            np.zeros((1, 22, 1)), np.ones((1, 22, 1)), np.zeros((1, 22, 1, 32)),
            np.ones((1, 22, 1, 32)), {}, {}, {"score": 0.7}, {})
        metadata = SourceMetadata("baseline-id", "src", "manifest", str(base_dir), 1, 0, "split",
            "fingerprint", artifact_hashes, {"inm/baselines/eegnet.py": "match", "inm/baselines/training.py": "match"})
        legacy_meta = replace(metadata, task_path=str(self.root / "legacy"))
        return TaskSources(1, 0, metadata, legacy_meta, baseline, legacy)

    def _run(self, sources=None, bundle=None):
        from unittest.mock import patch
        with patch("inm.encoder_audit.alignment._load_source_bundle", return_value=bundle or self.bundle):
            return audit_checkpoints(sources or self.sources, self.aligned, self.cfg)

    def test_all_three_arms_replay_clean_history_reload_and_frozen_state(self):
        report = self._run()
        self.assertEqual(set(report["arms"]), {"eegnet_reference__full", "masked_eegnet__full", "masked_eegnet__mixed"})
        self.assertEqual(report["partitions"], ["train", "validation"])
        self.assertTrue(report["no_test_rows_used"])
        for result in report["arms"].values():
            self.assertEqual(set(result["metrics"]), {"train", "validation"})
            self.assertTrue(result["model_state_unchanged"])
            self.assertEqual(result["reload_probability_max_abs_error"], {"train": 0.0, "validation": 0.0})
            self.assertEqual(result["optimization_time_label"], "training_mode_with_epoch_masks_and_dropout")
            self.assertIn("balanced_accuracy", result["metrics"]["validation"])
            self.assertIn("confusion_matrix", result["metrics"]["validation"])

    def test_wrong_selected_epoch_fails(self):
        sources = self._make_sources({"train": np.arange(0, 4), "validation": np.arange(4, 8)}, bad_epoch=3)
        with self.assertRaisesRegex(CheckpointAuditError, "epoch"):
            self._run(sources=sources)

    def test_wrong_class_order_fails(self):
        sources = self._make_sources({"train": np.arange(0, 4), "validation": np.arange(4, 8)}, bad_class_order=True)
        with self.assertRaisesRegex(CheckpointAuditError, "class or channel order"):
            self._run(sources=sources)

    def test_reordered_or_mislabeled_ids_fail(self):
        wrong = PartitionView(tuple(reversed(self.sources.baseline.train.sample_ids)),
                              self.sources.baseline.train.labels.copy())
        broken = replace(self.sources, baseline=replace(self.sources.baseline, train=wrong))
        with self.assertRaisesRegex(CheckpointAuditError, "labels do not match stable"):
            self._run(sources=broken)

    def test_selected_history_metric_mismatch_fails(self):
        arm = "masked_eegnet__mixed"
        checkpoint = self.sources.baseline.checkpoints[arm]
        row = copy.deepcopy(checkpoint.history_selection[0])
        row["clean_validation_metrics"]["accuracy"] = 0.123
        changed = replace(checkpoint, history_selection=(row,))
        items = dict(self.sources.baseline.checkpoints); items[arm] = changed
        broken = replace(self.sources, baseline=replace(self.sources.baseline, checkpoints=items))
        with self.assertRaisesRegex(CheckpointAuditError, "accuracy differs"):
            self._run(sources=broken)

    def test_nonfinite_evaluated_input_fails(self):
        bad = self.x.copy(); bad[0, 0, 0] = np.nan
        bundle = SimpleNamespace(**{**vars(self.bundle), "x": bad})
        with self.assertRaisesRegex(CheckpointAuditError, "train source contains nonfinite"):
            self._run(bundle=bundle)

    def test_test_rows_cannot_affect_outputs(self):
        first = self._run()
        changed_x = self.x.copy(); changed_x[8:] = np.nan
        changed_y = self.y.copy(); changed_y[8:] = 999
        poisoned = SimpleNamespace(**{**vars(self.bundle), "x": changed_x, "y": changed_y})
        second = self._run(bundle=poisoned)
        self.assertEqual(first, second)

    def test_historical_source_mismatch_blocks_replay(self):
        md = replace(self.sources.baseline_metadata, replay_source_status={"inm/baselines/eegnet.py": "mismatch"})
        blocked = replace(self.sources, baseline_metadata=md)
        with self.assertRaisesRegex(CheckpointAuditError, "inm/baselines/eegnet.py"):
            self._run(sources=blocked)


if __name__ == "__main__":
    unittest.main()
