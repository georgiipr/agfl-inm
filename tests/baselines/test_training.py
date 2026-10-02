import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from inm.baselines.data import PreparedData
from inm.baselines.training import fit_classifier, load_classifier, _selection_evidence


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def constructor_settings(self):
        return {}

    def forward(self, raw, mask=None):
        observed = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        value = observed.mean((1, 2, 3)).unsqueeze(1)
        return self.linear(value)


def prepared_fixture():
    rng = np.random.default_rng(91)
    labels = np.tile(np.arange(4), 4)
    signals = rng.normal(size=(16, 22, 1000)).astype(np.float32)
    for i, label in enumerate(labels):
        signals[i, label::4] += label * 0.1
    return PreparedData(
        signals=signals, filtered_signals=signals.copy(), labels=labels,
        sample_ids=tuple(f"trial-{i}" for i in range(16)),
        split_indices={"train": np.arange(8), "validation": np.arange(8, 12),
                       "test": np.arange(12, 16)},
        channel_names=tuple(f"C{i}" for i in range(22)),
        metadata={"seed": 3, "subject": 1, "split_id": "split-a",
                  "label_names": ["a", "b", "c", "d"], "synthetic": True},
        normalization_stats={"mean": [0.0] * 22, "std": [1.0] * 22},
        data_fingerprint="fixture-data")


def config(policy="full", epochs=4, minimum=1, patience=0):
    return {"protocol": "within_session", "protocol_description": "synthetic T split",
            "config_path": None,
            "preprocessing": {"channels": 22, "windows": 4, "window_samples": 250},
            "training": {"epochs": epochs, "batch_size": 4, "learning_rate": .001,
                         "weight_decay": .0001, "warmup_epochs": 0, "gradient_clip": 1.,
                         "minimum_epochs": minimum, "patience": patience},
            "selection": {"policy": policy}}


def evidence(score, loss):
    return {"policy": "full", "score": score, "tie_break_log_loss": loss,
            "validation_banks": {"full": {"balanced_accuracy": score,
                                               "log_loss": loss, "mask_sha256": "full"}}}


class BaselineTrainingTests(unittest.TestCase):
    def test_best_checkpoint_survives_worse_later_epoch_and_reloads(self):
        data = prepared_fixture()
        stats_before = copy.deepcopy(data.normalization_stats)
        schedule = [evidence(.75, .5), evidence(.5, .4), evidence(.5, .3), evidence(.25, .2)]
        with tempfile.TemporaryDirectory() as temp, patch(
                "inm.baselines.training._selection_evidence", side_effect=schedule):
            result = fit_classifier(TinyClassifier(), data, config(epochs=4),
                                    "eegnet_reference__full", temp)
            self.assertEqual(result["selected_epoch"], 1)
            reloaded, payload = load_classifier(result["checkpoint_path"],
                                                 model_factory=TinyClassifier)
            self.assertEqual(payload["metadata"]["selected_epoch"], 1)
            self.assertEqual(payload["metadata"]["normalization_stats"], stats_before)
            val = torch.from_numpy(data.signals[8:9].reshape(1, 22, 4, 250))
            mask = torch.ones(1, 22, 4, dtype=torch.bool)
            torch.testing.assert_close(result["model"](val, mask), reloaded(val, mask))
            self.assertEqual(data.normalization_stats, stats_before)

    def test_tie_break_uses_validation_log_loss(self):
        data = prepared_fixture()
        schedule = [evidence(.5, .8), evidence(.5, .4), evidence(.5, .6)]
        with tempfile.TemporaryDirectory() as temp, patch(
                "inm.baselines.training._selection_evidence", side_effect=schedule):
            result = fit_classifier(TinyClassifier(), data, config(epochs=3), "model__full", temp)
        self.assertEqual(result["selected_epoch"], 2)

    def test_robust_selection_is_exact_equal_mean_of_five_fixed_banks(self):
        scores = iter([.2, .4, .6, .8, 1.0])
        losses = iter([1.0, .8, .6, .4, .2])
        data = prepared_fixture()
        x = torch.from_numpy(data.signals[8:12].reshape(4, 22, 4, 250))
        labels = data.labels[8:12]

        def fake_eval(model, x, labels, mask, batch_size, device):
            return {"balanced_accuracy": next(scores), "log_loss": next(losses)}

        with patch("inm.baselines.training._eval", side_effect=fake_eval):
            result = _selection_evidence(TinyClassifier(), x, labels, config("robust"),
                data.sample_ids[8:12], 3, 1, 4, torch.device("cpu"))
        self.assertAlmostEqual(result["score"], .6)
        self.assertAlmostEqual(result["tie_break_log_loss"], .6)
        self.assertEqual(list(result["validation_banks"]),
                         ["full", "random_static_16", "dynamic_random_16",
                          "random_static_6", "dynamic_random_6"])

    def test_test_labels_do_not_affect_selection_or_training(self):
        first, second = prepared_fixture(), prepared_fixture()
        second.labels[second.split_indices["test"]] = [3, 2, 1, 0]
        torch.manual_seed(47)
        initial = TinyClassifier().state_dict()
        model_a, model_b = TinyClassifier(), TinyClassifier()
        model_a.load_state_dict(initial)
        model_b.load_state_dict(initial)
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            a = fit_classifier(model_a, first, config(epochs=2), "model__full", one)
            b = fit_classifier(model_b, second, config(epochs=2), "model__full", two)
        self.assertEqual(a["selected_epoch"], b["selected_epoch"])
        for key, value in a["model"].state_dict().items():
            torch.testing.assert_close(value, b["model"].state_dict()[key])

    def test_minimum_epoch_and_patience_are_respected(self):
        data = prepared_fixture()
        plateau = [evidence(.5, .5) for _ in range(8)]
        with tempfile.TemporaryDirectory() as temp, patch(
                "inm.baselines.training._selection_evidence", side_effect=plateau):
            result = fit_classifier(TinyClassifier(), data,
                config(epochs=8, minimum=4, patience=2), "model__full", temp)
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(result["epochs_trained"], 4)

    def test_missing_training_class_fails_visibly(self):
        data = prepared_fixture()
        data.labels[data.split_indices["train"]] = 0
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "training partition"):
            fit_classifier(TinyClassifier(), data, config(epochs=1), "model__full", temp)


if __name__ == "__main__":
    unittest.main()
