"""Numerical and boundary checks for covariance completion."""

import unittest

import numpy as np
import tensorflow as tf

from published_covariance.completion import CovarianceCompleter, hidden_mse, zero_fill


class CompletionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(811)
        factor = rng.normal(size=(22, 22))
        self.second_moment = factor @ factor.T / 22 + np.eye(22) * .2
        self.x = rng.normal(size=(2, 22, 13)).astype(np.float32)
        self.mask = np.ones((2, 22), dtype=bool)
        self.mask[0, 15:] = False
        self.mask[1, 6:] = False

    def test_native_length_solve_and_exact_observations(self):
        completer = CovarianceCompleter(self.second_moment)
        actual = completer.complete(self.x, self.mask).numpy()
        sigma = self.second_moment + np.eye(22) * 1e-6
        for row in range(2):
            observed = np.flatnonzero(self.mask[row])
            hidden = np.flatnonzero(~self.mask[row])
            ridge = .001 * np.diag(sigma).mean()
            expected = sigma[np.ix_(hidden, observed)] @ np.linalg.solve(
                sigma[np.ix_(observed, observed)] + ridge * np.eye(len(observed)),
                self.x[row, observed].astype(np.float64))
            np.testing.assert_allclose(actual[row, hidden], expected, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(actual[self.mask], self.x[self.mask])
        np.testing.assert_array_equal(completer.complete(self.x, np.ones_like(self.mask)), self.x)

    def test_hidden_nonfinite_sanitization_and_differentiable_empty_loss(self):
        completer = CovarianceCompleter(self.second_moment)
        expected = completer.complete(self.x, self.mask).numpy()
        altered = self.x.copy()
        altered[~self.mask] = np.nan
        np.testing.assert_array_equal(completer.complete(altered, self.mask), expected)
        np.testing.assert_array_equal(zero_fill(altered, self.mask),
                                      np.where(self.mask[:, :, None], self.x, 0))
        with tf.GradientTape() as tape:
            pred = tf.Variable(self.x)
            loss = hidden_mse(pred, pred, np.ones_like(self.mask))
        grad = tape.gradient(loss, pred)
        self.assertEqual(float(loss), 0.)
        self.assertIsNotNone(grad)

    def test_hidden_mse_uses_hidden_targets_only(self):
        target = np.full((1, 22, 2), 10_000., dtype=np.float64)
        prediction = np.full_like(target, -20_000.)
        mask = np.ones((1, 22), dtype=bool)
        mask[0, 0] = False
        target[0, 0] = [3., 7.]
        prediction[0, 0] = [4., 8.]
        self.assertEqual(float(hidden_mse(prediction, target, mask)), 1.)

    def test_covariance_gradient_is_finite(self):
        completer = CovarianceCompleter(self.second_moment)
        with tf.GradientTape() as tape:
            out = completer.complete(self.x, self.mask)
            loss = tf.reduce_mean(tf.square(tf.cast(out, tf.float64)))
        grad = tape.gradient(loss, completer.free)
        self.assertTrue(np.isfinite(grad.numpy()).all())
        self.assertGreater(float(tf.linalg.global_norm([grad])), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
