"""Synthetic CPU acceptance checks for candidate probes and robustness rows."""
from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from inm.encoder_candidates import diagnostics
from inm.encoder_candidates.models import build_model


@dataclass
class Part:
    raw: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]


@dataclass
class Prepared:
    subject: int
    seed: int
    train: Part
    validation: Part
    normalization: dict
    split_id: str
    data_id: str
    provenance: dict


def _prepared(n=8):
    rng = np.random.default_rng(71)
    labels = np.arange(n, dtype=np.int64) % 4
    train = rng.normal(size=(n, 22, 4, 250)).astype(np.float32) * .01
    validation = rng.normal(size=(n, 22, 4, 250)).astype(np.float32) * .01
    # Same local class cue, without making the fixture a model-quality claim.
    train[:, 0, 0, 0] += labels
    validation[:, 0, 0, 0] += labels
    return Prepared(2, 5,
        Part(train, labels.copy(), tuple(f"train-{i}" for i in range(n))),
        Part(validation, labels.copy(), tuple(f"validation-{i}" for i in range(n))),
        {"mean": [0.] * 22, "std": [1.] * 22}, "split-id", "data-id", {})


def _cfg():
    return {"training": {"batch_size": 4},
        "probes": {"C": 1., "max_iter": 1000, "solver": "lbfgs", "tol": 1e-4},
        "conditions": [
            {"name": "full_22", "pattern": "full", "retained": 22, "repeats": 1},
            {"name": "random_static_16", "pattern": "random_static", "retained": 16, "repeats": 5},
            {"name": "dynamic_random_16", "pattern": "dynamic_random", "retained": 16, "repeats": 5},
            {"name": "random_static_6", "pattern": "random_static", "retained": 6, "repeats": 5},
            {"name": "dynamic_random_6", "pattern": "dynamic_random", "retained": 6, "repeats": 5}],
        "synthetic": True}


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_features_are_frozen_and_channel_window_local(self):
        model = build_model("local_power", seed=4).eval()
        model.freeze_features()
        raw = torch.randn(2, 22, 4, 250)
        mask = torch.ones((2, 22, 4), dtype=torch.bool)
        with torch.no_grad():
            original = model.forward_features(raw, mask)
            changed = raw.clone()
            changed[:, 3, 2] += 2
            perturbed = model.forward_features(changed, mask)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(torch.equal(original[:, 3, 2], perturbed[:, 3, 2]))
        untouched = torch.ones((22, 4), dtype=torch.bool)
        untouched[3, 2] = False
        self.assertTrue(torch.equal(original[:, untouched], perturbed[:, untouched]))

    def test_two_probes_use_train_scaling_preserve_true_labels_and_reload(self):
        data = _prepared()
        model = build_model("local_control", seed=8)
        with tempfile.TemporaryDirectory() as temp:
            output = diagnostics.fit_local_probes(model, data, _cfg(), temp)
            self.assertEqual(set(output), {"probe", "probe_shuffled"})
            self.assertEqual(len(list(Path(temp).glob("*.npz"))), 2)
            with np.load(Path(temp) / "probe.npz", allow_pickle=False) as plain, \
                    np.load(Path(temp) / "probe_shuffled.npz", allow_pickle=False) as shuffled:
                self.assertTrue(np.array_equal(plain["train_labels_true"], data.train.labels))
                self.assertTrue(np.array_equal(shuffled["train_labels_true"], data.train.labels))
                self.assertFalse(np.array_equal(shuffled["train_labels_used"], data.train.labels))
                self.assertTrue(np.array_equal(plain["train_labels_used"], data.train.labels))
                self.assertTrue(np.allclose(plain["scaler_mean"],
                    np.load(Path(temp) / "probe.npz", allow_pickle=False)["scaler_mean"]))
                np.testing.assert_allclose(plain["validation_probabilities"],
                    plain["validation_reloaded_probabilities"], rtol=1e-4, atol=1e-5)
                np.testing.assert_allclose(shuffled["validation_probabilities"],
                    shuffled["validation_reloaded_probabilities"], rtol=1e-4, atol=1e-5)
                self.assertEqual(plain["validation_probabilities"].shape, (8, 4))
                self.assertTrue(bool(plain["converged"]))
                self.assertTrue(np.isfinite(output["probe"]["validation_metrics"]["log_loss"]))
            # Two fixed fits per local arm, four total for a candidate task.
            power_dir = Path(temp) / "local_power"
            power_dir.mkdir()
            power = diagnostics.fit_local_probes(
                build_model("local_power", seed=8), data, _cfg(), power_dir)
            self.assertEqual(set(power), {"probe", "probe_shuffled"})
            self.assertEqual(len(list(Path(temp).rglob("*.npz"))), 4)

    def test_probe_scaler_is_fit_on_training_rows_only(self):
        train = np.arange(4 * 2, dtype=np.float32).reshape(4, 1, 1, 2)
        validation = np.full((4, 1, 1, 2), 10000, dtype=np.float32)
        y = np.arange(4, dtype=np.int64)
        result = diagnostics._probe_arrays(train, validation, y, y,
            tuple(f"t{i}" for i in range(4)), tuple(f"v{i}" for i in range(4)), 0, False)
        self.assertTrue(np.allclose(result["arrays"]["scaler_mean"], train.reshape(4, -1).mean(0)))
        self.assertLess(float(result["arrays"]["scaler_mean"].max()), 100)

    def test_nonconvergence_is_reported_without_erasing_scores(self):
        data = _prepared()
        real_fit = diagnostics.LogisticRegression.fit
        def nonconverged(estimator, x, y):
            real_fit(estimator, x, y)
            estimator.n_iter_ = np.asarray([estimator.max_iter], dtype=np.int32)
            return estimator
        with patch.object(diagnostics.LogisticRegression, "fit", nonconverged), \
                tempfile.TemporaryDirectory() as temp:
            result = diagnostics.fit_local_probes(
                build_model("local_power", seed=2), data, _cfg(), temp)
        self.assertFalse(result["probe"]["converged"])
        self.assertTrue(np.isfinite(result["probe"]["validation_metrics"]["log_loss"]))

    def test_paired_masks_are_stable_and_spatial_models_receive_raw_masks(self):
        data = _prepared()
        cfg = _cfg()
        model = build_model("spatial_eegnet", seed=3).eval()
        rows = diagnostics.evaluate_selected(model, data, cfg)
        again = diagnostics.evaluate_selected(model, data, cfg)
        local_rows = diagnostics.evaluate_selected(
            build_model("local_control", seed=3).eval(), data, cfg)
        self.assertEqual(len(rows), 21)
        self.assertEqual([(r["scenario"], r["repeat"]) for r in rows],
                         [(r["scenario"], r["repeat"]) for r in again])
        self.assertEqual([r["mask_sha256"] for r in rows],
                         [r["mask_sha256"] for r in again])
        self.assertEqual([r["mask_sha256"] for r in rows],
                         [r["mask_sha256"] for r in local_rows])
        self.assertTrue(all(np.isfinite(r["log_loss"]) and
                            np.isfinite(r["balanced_accuracy"]) for r in rows))
        self.assertEqual(rows[0]["scenario"], "full_22")
        self.assertEqual(rows[-1]["repeat"], 4)

    def test_hidden_nan_and_all_missing_mask_failures_are_mask_safe(self):
        raw = torch.zeros((1, 22, 4, 250))
        mask = torch.ones((1, 22, 4), dtype=torch.bool)
        mask[:, 0, :] = False
        raw[:, 0] = float("nan")
        probabilities = diagnostics._mask_model(build_model("spatial_filterbank", 1),
            raw.numpy(), mask.numpy(), 1, torch.device("cpu"))
        self.assertTrue(np.isfinite(probabilities).all())
        mask[:, :, :] = False
        with self.assertRaisesRegex(ValueError, "observed electrode"):
            diagnostics._mask_model(build_model("spatial_filterbank", 1), raw.numpy(),
                mask.numpy(), 1, torch.device("cpu"))

    def test_spatial_mixing_responds_to_observed_raw_channel(self):
        model = build_model("spatial_eegnet", seed=13).eval()
        raw = torch.randn(1, 22, 4, 250)
        full = torch.ones((1, 22, 4), dtype=torch.bool)
        mask = full.clone()
        mask[:, 0, :] = False
        changed = raw.clone()
        changed[:, 1] += 4
        with torch.no_grad():
            full_features = model.forward_features(raw, full)
            masked_features = model.forward_features(raw, mask)
            changed_features = model.forward_features(changed, mask)
        self.assertFalse(torch.allclose(full_features, masked_features))
        self.assertFalse(torch.allclose(masked_features, changed_features))
        # EEGNet activations have 16 latent features; the original 22-channel
        # identity mask cannot be applied as if these were electrode features.
        self.assertEqual(masked_features.shape[-1], 16)
        self.assertNotEqual(mask.shape[1], masked_features.shape[-1])


if __name__ == "__main__":
    unittest.main()
