"""Synthetic CPU acceptance checks for candidate classifier fitting/reload."""
from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from inm.encoder_candidates import training
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
    rng = np.random.default_rng(9)
    labels = np.arange(n) % 4
    # A simple class signal in one raw sample makes this a learnability fixture.
    train = rng.normal(0, .01, (n, 22, 4, 250)).astype(np.float32)
    val = rng.normal(0, .01, (n, 22, 4, 250)).astype(np.float32)
    for i, label in enumerate(labels):
        train[i, 0, 0, 0] = float(label)
        val[i, 0, 0, 0] = float(label)
    return Prepared(1, 3,
        Part(train, labels.copy(), tuple(f"tr-{i}" for i in range(n))),
        Part(val, labels.copy(), tuple(f"va-{i}" for i in range(n))),
        {"mean": [0.0] * 22, "std": [1.0] * 22}, "split-x", "data-y",
        {"channel_order": [f"ch{i}" for i in range(22)]})


def _cfg(epochs=2, minimum=1, patience=0, batch=4):
    return {"synthetic": True, "training_regime": "full",
            "training": {"learning_rate": .001, "weight_decay": .0001,
                         "batch_size": batch, "maximum_epochs": epochs,
                         "minimum_epochs": minimum, "patience": patience,
                         "gradient_clip_norm": 1.0,
                         "warmup_epochs": min(1, max(0, epochs - 1))}}


class _TinyModel(nn.Module):
    def __init__(self, extra=0):
        super().__init__()
        self.linear = nn.Linear(1, 4)
        self.extra = nn.Parameter(torch.zeros(extra)) if extra else None
        self.seen = []

    def constructor_settings(self):
        return {"extra": 0 if self.extra is None else self.extra.numel()}

    def forward(self, raw, mask=None):
        if self.training:
            self.seen.extend(raw[:, 0, 0, 0].detach().cpu().tolist())
        return self.linear(raw[:, 0, 0, 0, None])

    def clip_weights(self):
        return None


class CandidateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_lr_warmup_and_cosine_have_exact_endpoints(self):
        settings = {"learning_rate": .001, "maximum_epochs": 250, "warmup_epochs": 10}
        self.assertAlmostEqual(training.learning_rate_for_epoch(1, settings), .0001)
        self.assertAlmostEqual(training.learning_rate_for_epoch(10, settings), .001)
        self.assertAlmostEqual(training.learning_rate_for_epoch(250, settings), 0.0, places=14)
        with self.assertRaises(ValueError):
            training.learning_rate_for_epoch(0, settings)

    def test_all_five_arms_save_and_reload_identical_probabilities(self):
        data = _prepared()
        cfg = _cfg(epochs=1, minimum=1, patience=0)
        # One optimizer update per fit: keeps all architecture checks tiny and CPU-only.
        cfg["training"]["batch_size"] = 8
        raw = torch.as_tensor(data.validation.raw)
        mask = torch.ones((len(raw), 22, 4), dtype=torch.bool)
        with tempfile.TemporaryDirectory() as temp:
            for arm in ("local_control", "local_power", "spatial_eegnet",
                        "spatial_filterbank", "spatial_transformer"):
                model = build_model(arm, seed=11)
                fitted = training.fit_classifier(model, data, cfg, arm,
                    Path(temp) / arm, device="cpu")
                self.assertEqual(fitted["status"], "complete")
                self.assertEqual(fitted["selected_epoch"], 1)
                fitted["model"].eval()
                with torch.no_grad():
                    probabilities = fitted["model"](raw, mask).softmax(-1)
                    # Reopen via the public restricted weights-only interface.
                    payload = torch.load(fitted["checkpoint_path"], map_location="cpu",
                                         weights_only=True)
                    restored = training.restore_model(arm, payload["constructor"],
                                                       payload["state_dict"]).eval()
                    restored_probabilities = restored(raw, mask).softmax(-1)
                torch.testing.assert_close(probabilities, restored_probabilities,
                                           rtol=1e-4, atol=1e-5)
                self.assertEqual(fitted["model"].training, False)
                with torch.no_grad():
                    torch.testing.assert_close(probabilities,
                        fitted["model"](raw, mask).softmax(-1), rtol=0, atol=0)

    def test_learnability_fixture_reduces_optimization_loss(self):
        data = _prepared(n=16)
        cfg = _cfg(epochs=12, minimum=12, patience=0, batch=16)
        with tempfile.TemporaryDirectory() as temp, patch.object(
                training, "restore_model", side_effect=lambda arm, ctor, state:
                _restore_tiny(ctor, state)):
            result = training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)
        losses = [row["loss"] for row in result["optimization"]]
        self.assertLess(losses[-1], losses[0])

    def test_nonfinite_input_and_corrupt_checkpoint_fail_closed(self):
        data = _prepared()
        cfg = _cfg(epochs=1, minimum=1)
        bad = _prepared()
        bad.train.raw[0, 0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temp, patch.object(
                training, "restore_model", side_effect=lambda arm, ctor, state:
                _restore_tiny(ctor, state)):
            with self.assertRaisesRegex(ValueError, "finite"):
                training.fit_classifier(_TinyModel(), bad, cfg, "tiny", temp)
            fit = training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)
            reused = training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)
            self.assertTrue(reused["reused"])
            changed_identity = _prepared()
            changed_identity.data_id = "different-data"
            with self.assertRaisesRegex(ValueError, "failed verification"):
                training.fit_classifier(_TinyModel(), changed_identity, cfg, "tiny", temp)
            Path(fit["checkpoint_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "failed verification"):
                training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)

    def test_parameter_count_does_not_change_private_batch_order(self):
        data = _prepared()
        cfg = _cfg(epochs=1, minimum=1, batch=2)
        models = [_TinyModel(0), _TinyModel(31)]
        with tempfile.TemporaryDirectory() as temp, patch.object(
                training, "restore_model", side_effect=lambda arm, ctor, state:
                _restore_tiny(ctor, state)):
            for i, model in enumerate(models):
                training.fit_classifier(model, data, cfg, f"tiny{i}", Path(temp) / str(i))
        self.assertEqual(models[0].seen, models[1].seen)
        self.assertEqual(len(models[0].seen), len(data.train.labels))

    def test_validation_ties_use_loss_then_keep_earlier_exact_tie(self):
        data = _prepared()
        cfg = _cfg(epochs=3, minimum=3, patience=0)
        scripted = [(.75, .9), (.75, .6), (.75, .6)]
        calls = {"count": 0}
        def fake_eval(model, x, labels, batch_size, device):
            # Fit loop calls validation then training. Final restored evaluations
            # repeat the winning evidence to verify that selected weights persisted.
            number = calls["count"]
            calls["count"] += 1
            idx = number // 2
            score, loss = scripted[min(idx, len(scripted) - 1)] if number % 2 == 0 else (.5, 1.0)
            if number >= 6:
                score, loss = (.5, 1.0) if number == 6 else (.75, .6)
            metrics = {"balanced_accuracy": score, "log_loss": loss}
            return metrics, np.full((len(labels), 4), .25)
        with tempfile.TemporaryDirectory() as temp, patch.object(
                training, "restore_model", side_effect=lambda arm, ctor, state:
                _restore_tiny(ctor, state)), patch.object(training, "_evaluate", side_effect=fake_eval):
            result = training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)
        self.assertEqual(result["selected_epoch"], 2)
        self.assertEqual(result["epochs_trained"], 3)

    def test_minimum_and_patience_boundary_and_synthetic_budget_gate(self):
        data = _prepared()
        cfg = _cfg(epochs=6, minimum=3, patience=2)
        calls = {"count": 0}
        def equal_eval(model, x, labels, batch_size, device):
            number = calls["count"]
            calls["count"] += 1
            m = {"balanced_accuracy": .5, "log_loss": 1.0}
            return m, np.full((len(labels), 4), .25)
        with tempfile.TemporaryDirectory() as temp, patch.object(
                training, "restore_model", side_effect=lambda arm, ctor, state:
                _restore_tiny(ctor, state)), patch.object(training, "_evaluate", side_effect=equal_eval):
            fit = training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)
        # Best at epoch 1, but minimum 3 and two stale epochs are both required.
        self.assertEqual(fit["selected_epoch"], 1)
        self.assertEqual(fit["epochs_trained"], 3)
        cfg["synthetic"] = False
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "fixed full training budget"):
                training.fit_classifier(_TinyModel(), data, cfg, "tiny", temp)


def _restore_tiny(constructor, state_dict):
    model = _TinyModel(constructor["extra"])
    model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    unittest.main()
