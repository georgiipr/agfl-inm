"""Training selection and covariance-only optimizer checks."""

import random
import unittest

import numpy as np
import tensorflow as tf

from published_covariance.completion import CovarianceCompleter
from published_covariance.training import select_checkpoint, train_covariance


class _ConstantFrozen:
    def __init__(self, samples):
        inp = tf.keras.Input(shape=(22, samples))
        pooled = tf.keras.layers.GlobalAveragePooling1D()(inp)
        out = tf.keras.layers.Dense(4, kernel_initializer="zeros", bias_initializer="zeros",
                                    activation="softmax")(pooled)
        self.model = tf.keras.Model(inp, out)
        self.model.trainable = False

    def __call__(self, values, training=False):
        return self.model(values, training=False)


class TrainingTests(unittest.TestCase):
    def test_checkpoint_ordering(self):
        records = [
            {"epoch": 1, "degraded_ba": .5, "degraded_log_loss": .6},
            {"epoch": 0, "degraded_ba": .5, "degraded_log_loss": .6},
            {"epoch": 2, "degraded_ba": .5, "degraded_log_loss": .4},
            {"epoch": 3, "degraded_ba": .4, "degraded_log_loss": .1},
        ]
        self.assertEqual(select_checkpoint(records)["epoch"], 2)

    def test_actual_optimizer_update_and_exact_epoch_zero_restore(self):
        rng = np.random.default_rng(29)
        train = rng.normal(size=(8, 22, 31)).astype(np.float32)
        validation = rng.normal(size=(8, 22, 31)).astype(np.float32)
        labels = np.arange(8, dtype=np.int64) % 4
        original_validation = validation.copy()
        py_state = random.getstate()
        np_state = np.random.get_state()
        classifier = _ConstantFrozen(31)
        frozen = [weight.numpy().copy() for weight in classifier.model.weights]
        second_moment = np.einsum("nct,ndt->cd", train.astype(np.float64),
                                  train.astype(np.float64)) / (len(train) * train.shape[2])
        initial = CovarianceCompleter(second_moment).free.numpy().copy()
        result = train_covariance(
            classifier, train, labels,
            [f"A01:T:r01:t{i:03d}" for i in range(8)],
            validation, labels,
            [f"A01:T:r02:t{i:03d}" for i in range(8)],
            "A01", 0, max_epochs=2, min_epochs=0, patience=1, synthetic=True)
        self.assertEqual(result["selected_epoch"], 0)
        self.assertGreater(result["history"][1]["parameter_delta_norm"], 0.)
        np.testing.assert_allclose(result["completer"].free.numpy(), initial, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(validation, original_validation)
        for before, after in zip(frozen, classifier.model.weights):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertEqual(after[0], np_state[0])
        np.testing.assert_array_equal(after[1], np_state[1])
        self.assertEqual(after[2:], np_state[2:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
