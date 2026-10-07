import unittest

import numpy as np
import tensorflow as tf

from published_covariance.classifier import FrozenClassifier


class FrozenClassifierTests(unittest.TestCase):
    def test_native_forward_input_gradient_and_frozen_state(self):
        classifier = FrozenClassifier("A01")
        x = tf.constant(np.random.default_rng(93).normal(size=(2, 22, 1125)).astype(np.float32))
        before = [value.numpy().copy() for value in classifier.model.weights]
        with tf.GradientTape() as tape:
            tape.watch(x)
            probabilities = classifier(x, training=True)
            objective = tf.reduce_sum(probabilities[:, 0])
        gradient = tape.gradient(objective, x).numpy()
        self.assertEqual(tuple(probabilities.shape), (2, 4))
        np.testing.assert_allclose(np.sum(probabilities.numpy(), axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(np.linalg.norm(gradient), 0)
        np.testing.assert_array_equal(probabilities, classifier(x, training=False))
        self.assertEqual(classifier.model.trainable_variables, [])
        for value, expected in zip(classifier.model.weights, before):
            np.testing.assert_array_equal(value, expected)

    def test_subject_shape_dtype_and_nonfinite_validation(self):
        with self.assertRaises(ValueError):
            FrozenClassifier("A10")
        classifier = FrozenClassifier("A01")
        for invalid in (
            tf.zeros((2, 22, 1124), tf.float32),
            tf.zeros((2, 21, 1125), tf.float32),
            tf.zeros((22, 1125), tf.float32),
            tf.zeros((2, 22, 1125), tf.int32),
            tf.fill((1, 22, 1125), tf.constant(np.nan, tf.float32)),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
                    classifier(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
