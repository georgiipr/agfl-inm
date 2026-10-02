import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inm.baselines.covariance import (CovarianceClassifier, covariance_features,
                                      covariance_log_matrix)


def tiny_signals(seed=5, samples_per_class=6, channels=4, time=80):
    rng = np.random.default_rng(seed)
    x, y = [], []
    for label in range(4):
        for _ in range(samples_per_class):
            trial = rng.normal(size=(channels, time))
            trial[label % channels] += rng.normal(scale=0.2, size=time)
            x.append(trial)
            y.append(label)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)


class CovarianceTests(unittest.TestCase):
    def test_near_singular_trials_produce_finite_symmetric_log_features(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=100)
        trial = np.stack([base, base + 1e-14 * rng.normal(size=100), np.ones(100)])
        logged = covariance_log_matrix(trial)
        features = covariance_features(trial[None])
        self.assertTrue(np.isfinite(logged).all())
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(logged, logged.T, rtol=0, atol=1e-12)

    def test_upper_triangle_weights_off_diagonal_by_sqrt_two(self):
        matrix = np.array([[2.0, 3.0, 5.0], [3.0, 7.0, 11.0], [5.0, 11.0, 13.0]])
        with patch("inm.baselines.covariance.covariance_log_matrix", return_value=matrix):
            actual = covariance_features(np.ones((1, 3, 4)))[0]
        expected = np.array([2.0, 3.0 * np.sqrt(2), 5.0 * np.sqrt(2),
                             7.0, 11.0 * np.sqrt(2), 13.0])
        np.testing.assert_allclose(actual, expected)

    def test_four_class_fit_probability_order_train_only_scaling_and_roundtrip(self):
        x, y = tiny_signals()
        model = CovarianceClassifier().fit(x, y)
        self.assertEqual(model.classes_.tolist(), [0, 1, 2, 3])
        self.assertEqual(model.predict_proba(x[:3]).shape, (3, 4))
        self.assertTrue(np.isfinite(model.predict_proba(x[:3])).all())
        train_mean = model.scaler_.mean_.copy()
        train_scale = model.scaler_.scale_.copy()
        held_out = np.full((3, x.shape[1], x.shape[2]), 1e8)
        model.predict_proba(held_out)
        np.testing.assert_array_equal(model.scaler_.mean_, train_mean)
        np.testing.assert_array_equal(model.scaler_.scale_, train_scale)
        with tempfile.TemporaryDirectory() as temp:
            path = model.save(Path(temp) / "covariance.npz")
            restored = CovarianceClassifier.load(path)
            np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x),
                                       rtol=1e-12, atol=1e-12)
            self.assertEqual(restored.classes_.tolist(), [0, 1, 2, 3])

    def test_scaler_fitted_from_training_features_only(self):
        x, y = tiny_signals(samples_per_class=4)
        train_x, train_y = x[:8], y[:8]
        # Every class must be represented; use the first two trials per class.
        train_indices = np.array([0, 1, 4, 5, 8, 9, 12, 13])
        model = CovarianceClassifier().fit(x[train_indices], y[train_indices])
        expected = covariance_features(x[train_indices]).mean(axis=0)
        np.testing.assert_allclose(model.scaler_.mean_, expected)


if __name__ == "__main__":
    unittest.main()
