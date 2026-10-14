"""Native-length zero fill and differentiable Gaussian covariance completion."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


_N_CHANNELS = 22
_LOWER = np.tril_indices(_N_CHANNELS)
_DIAGONAL_FREE = np.cumsum(np.arange(1, _N_CHANNELS + 1)) - 1


def _validated_values_and_mask(values, mask):
    x = tf.convert_to_tensor(values)
    if not x.dtype.is_floating:
        raise TypeError("values must have a floating-point dtype")
    m = tf.convert_to_tensor(mask)
    if m.dtype != tf.bool:
        raise TypeError("mask must have Boolean dtype")
    checks = [
        tf.debugging.assert_rank(x, 3, message="values must have shape [B,22,T]"),
        tf.debugging.assert_positive(tf.shape(x)[0], message="batch cannot be empty"),
        tf.debugging.assert_equal(tf.shape(x)[1], _N_CHANNELS, message="expected 22 channels"),
        tf.debugging.assert_positive(tf.shape(x)[2], message="time dimension cannot be empty"),
        tf.debugging.assert_rank(m, 2, message="mask must have shape [B,22]"),
        tf.debugging.assert_equal(tf.shape(m), tf.shape(x)[:2], message="mask shape mismatch"),
        tf.debugging.assert_equal(tf.reduce_all(tf.reduce_any(m, axis=1)), True,
                                  message="all-missing trials are not allowed"),
        tf.debugging.assert_all_finite(tf.boolean_mask(x, m),
                                       "observed values contain NaN or infinity"),
    ]
    with tf.control_dependencies(checks):
        return tf.identity(x), tf.identity(m)


def zero_fill(values, mask):
    """Return values with unobserved electrodes replaced by zero, safely."""
    x, m = _validated_values_and_mask(values, mask)
    # Selection precedes arithmetic so hidden NaN/Inf never enter the result.
    return tf.where(m[:, :, None], x, tf.zeros_like(x))


def hidden_mse(prediction, target, mask):
    """Mean squared error over hidden samples, with a differentiable empty zero."""
    pred = tf.convert_to_tensor(prediction)
    x, m = _validated_values_and_mask(target, mask)
    checks = [tf.debugging.assert_equal(tf.shape(pred), tf.shape(x)),
              tf.debugging.assert_all_finite(pred, "prediction contains NaN or infinity"),
              tf.debugging.assert_all_finite(x, "hidden-loss target contains NaN or infinity")]
    with tf.control_dependencies(checks):
        hidden = ~m[:, :, None]
        safe_target = tf.where(hidden, x, tf.zeros_like(x))
        safe_prediction = tf.where(hidden, pred, tf.zeros_like(pred))
        delta = tf.cast(safe_prediction, tf.float64) - tf.cast(safe_target, tf.float64)
        count = tf.reduce_sum(tf.cast(~m, tf.float64)) * tf.cast(tf.shape(x)[2], tf.float64)
        numerator = tf.reduce_sum(tf.square(delta))
        return tf.math.divide_no_nan(numerator, count)


class CovarianceCompleter:
    """Conditional Gaussian completion from one training second moment.

    ``free`` stores the 253 lower-triangle entries of a float64 Cholesky factor.
    The diagonal is represented by softplus plus ``1e-6``. ``complete`` accepts
    arbitrary native time lengths and preserves observed entries exactly.
    """

    def __init__(self, second_moment, trainable: bool = True):
        if not isinstance(trainable, (bool, np.bool_)):
            raise TypeError("trainable must be Boolean")
        s = np.asarray(second_moment, dtype=np.float64)
        if s.shape != (_N_CHANNELS, _N_CHANNELS) or not np.isfinite(s).all():
            raise ValueError("second_moment must be a finite 22x22 matrix")
        if not np.allclose(s, s.T, atol=1e-10, rtol=1e-10):
            raise ValueError("second_moment must be symmetric")
        sigma = s + np.eye(_N_CHANNELS, dtype=np.float64) * 1e-6
        try:
            chol = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            raise ValueError("second_moment plus jitter must be positive definite") from exc
        raw = chol[_LOWER].copy()
        # Inverse softplus of (diag(L) - epsilon), stable for large values.
        diagonal = np.maximum(raw[_DIAGONAL_FREE] - 1e-6, np.finfo(np.float64).tiny)
        raw[_DIAGONAL_FREE] = diagonal + np.log(-np.expm1(-diagonal))
        self.free = tf.Variable(raw, trainable=bool(trainable), dtype=tf.float64,
                                name="covariance_cholesky_free")
        self.initial_free = tf.constant(raw, dtype=tf.float64)
        self.trainable = bool(trainable)

    @property
    def covariance(self):
        lower = tf.scatter_nd(tf.constant(np.stack(_LOWER, axis=1), dtype=tf.int32),
                              self.free, [_N_CHANNELS, _N_CHANNELS])
        diag = tf.nn.softplus(tf.gather(self.free, _DIAGONAL_FREE)) + 1e-6
        lower = tf.linalg.set_diag(lower, diag)
        return tf.matmul(lower, lower, transpose_b=True)

    def complete(self, values, mask):
        x, m = _validated_values_and_mask(values, mask)
        # The all-observed path is exactly the caller's original tensor.
        if tf.executing_eagerly() and bool(tf.reduce_all(m).numpy()):
            return x
        sigma = self.covariance
        sigma = tf.cast(sigma, tf.float64)
        # Safe selection happens before casting, solving, or multiplying.
        safe = tf.where(m[:, :, None], x, tf.zeros_like(x))
        safe64 = tf.cast(safe, tf.float64)
        obs = tf.cast(m, tf.float64)
        hidden = 1.0 - obs
        outer = obs[:, :, None] * obs[:, None, :]
        rho = 0.001 * tf.reduce_mean(tf.linalg.diag_part(sigma))
        system = sigma[None, :, :] * outer
        system += tf.linalg.diag(hidden + rho * obs)
        rhs = safe64
        solved = tf.linalg.solve(system, rhs)
        conditional = tf.matmul(sigma[None, :, :], solved)
        # Select the original observed values directly, preserving their bits.
        filled = tf.cast(conditional, x.dtype)
        return tf.where(m[:, :, None], x, filled)
