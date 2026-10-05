"""Fixed-probe synthetic acceptance; no recordings or real research fits."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

from inm.encoder_audit.probes import ProbeError, fit_probes, load_probe
from inm.tensor_attention import Tucker2


@dataclass
class Part:
    sample_ids: tuple[str, ...]
    labels: np.ndarray


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(29)
        model = Tucker2(22, 32, rank_channels=4, rank_features=4, ridge=.001)
        with torch.no_grad():
            model.U.copy_(torch.tensor(rng.normal(size=(22, 4)), dtype=torch.float32))
            model.V.copy_(torch.tensor(rng.normal(size=(32, 4)), dtype=torch.float32))
            model.fitted.fill_(True)
            model.fit_steps.fill_(1)
            model.training_seen_counts.fill_(8)
        self.factor_state = {key: value.clone() for key, value in model.state_dict().items()}
        train_y = np.tile(np.arange(4, dtype=np.int64), 3)
        validation_y = np.tile(np.arange(4, dtype=np.int64), 2)
        train_x = rng.normal(0, .04, size=(12, 22, 4, 32)).astype(np.float32)
        # A linearly separable class signal in one ordered feature coordinate.
        train_x[:, 0, 0, 0] += np.repeat(np.arange(4), 3) * 4.0
        # Validation is deliberately label-independent: all features are the
        # same while labels cover all classes, so labels cannot leak into fit.
        validation_x = np.zeros((8, 22, 4, 32), dtype=np.float32)
        self.ids_train = tuple(f"train-{i}" for i in range(12))
        self.ids_validation = tuple(f"validation-{i}" for i in range(8))
        metadata = lambda study, split: SimpleNamespace(study_id=study, split_id=split,
            source_sha256="a" * 64, manifest_sha256="b" * 64, artifact_sha256={"fixture": "c" * 64})
        self.sources = SimpleNamespace(subject=1, seed=2,
            baseline_metadata=metadata("baseline-id", "base-split"),
            legacy_metadata=metadata("legacy-id", "legacy-split"),
            legacy=SimpleNamespace(train=Part(self.ids_train, train_y),
                validation=Part(self.ids_validation, validation_y), train_features=train_x,
                validation_features=validation_x, factor_state=self.factor_state))
        self.train_x = train_x
        self.validation_x = validation_x

    def fit(self):
        return fit_probes(self.sources, {}, Path(self.tmp.name) / "probes")

    def test_fixed_probes_learn_and_save_reload_probabilities(self):
        before = {key: value.clone() for key, value in self.sources.legacy.factor_state.items()}
        result = self.fit()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["probes"]), {"features_ordered", "features_window_mean",
            "core_ordered", "features_ordered_shuffled"})
        ordered = result["probes"]["features_ordered"]
        self.assertGreater(ordered["train"]["balanced_accuracy"], .95)
        self.assertEqual(ordered["validation"]["balanced_accuracy"], .25)
        self.assertLessEqual(ordered["probability_reload_max_abs_error"], 1e-7)
        probe = load_probe(ordered["path"])
        self.assertEqual(probe.metadata["feature"]["width"], 2816)
        self.assertEqual(probe.metadata["sample_ids"]["validation"], list(self.ids_validation))
        for key in before:
            torch.testing.assert_close(before[key], self.sources.legacy.factor_state[key], rtol=0, atol=0)
        with np.load(ordered["path"], allow_pickle=False) as data:
            np.testing.assert_allclose(data["validation_probabilities"],
                probe.predict_proba(self.validation_x), rtol=1e-4, atol=1e-5)
            self.assertEqual(tuple(data["validation_ids"].tolist()), self.ids_validation)

    def test_scaler_uses_training_only_and_shuffle_touches_only_fit_labels(self):
        import sklearn.linear_model
        seen_labels = []
        original_fit = sklearn.linear_model.LogisticRegression.fit
        def capture_labels(model, x, y, *args, **kwargs):
            seen_labels.append(np.asarray(y).copy())
            return original_fit(model, x, y, *args, **kwargs)
        with patch("sklearn.linear_model.LogisticRegression.fit", new=capture_labels):
            result = self.fit()["probes"]
        ordered = result["features_ordered"]
        with np.load(ordered["path"], allow_pickle=False) as data:
            train_view = self.train_x.reshape(12, -1).astype(np.float64)
            np.testing.assert_allclose(data["mean"], train_view.mean(axis=0))
            shuffled = result["features_ordered_shuffled"]["path"]
        with np.load(shuffled, allow_pickle=False) as data:
            metadata = __import__("json").loads(str(data["metadata"].item()))
            self.assertEqual(metadata["fit"]["shuffle_seed"], 700003)
            self.assertEqual(metadata["fit"]["fit_labels"], "shuffled_train")
            np.testing.assert_array_equal(data["train_labels"], self.sources.legacy.train.labels)
            np.testing.assert_array_equal(data["validation_labels"], self.sources.legacy.validation.labels)
        expected = np.random.default_rng(700003).permutation(self.sources.legacy.train.labels)
        self.assertEqual(len(seen_labels), 4)
        for observed in seen_labels[:3]:
            np.testing.assert_array_equal(observed, self.sources.legacy.train.labels)
        np.testing.assert_array_equal(seen_labels[3], expected)
        self.assertFalse(np.array_equal(seen_labels[3], self.sources.legacy.train.labels))

    def test_invalid_labels_nonfinite_features_and_nonconvergence_fail(self):
        self.sources.legacy.validation.labels[0] = 9
        with self.assertRaisesRegex(ProbeError, "labels"):
            self.fit()
        self.sources.legacy.validation.labels[:] = np.tile(np.arange(4), 2)
        self.sources.legacy.train_features[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ProbeError, "finite"):
            self.fit()

        self.sources.legacy.train_features[:] = np.nan_to_num(self.sources.legacy.train_features)
        original_fit = __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression.fit
        def nonconverged(model, x, y, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("always", ConvergenceWarning)
                warnings.warn("synthetic nonconvergence", ConvergenceWarning)
            return original_fit(model, x, y, *args, **kwargs)
        with patch("sklearn.linear_model.LogisticRegression.fit", new=nonconverged):
            with self.assertRaisesRegex(ProbeError, "did not converge"):
                self.fit()

    def test_probe_matrix_rejects_reordered_or_nonfinite_inputs(self):
        probe = self.fit()["probes"]["features_ordered"]["probe"]
        with self.assertRaisesRegex(ProbeError, "expects normalized"):
            probe.predict_proba(self.validation_x[:, :, :, :31])
        bad = self.validation_x.copy()
        bad[0, 0, 0, 0] = np.inf
        with self.assertRaisesRegex(ProbeError, "nonfinite"):
            probe.predict_proba(bad)


if __name__ == "__main__":
    unittest.main()
