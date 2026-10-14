"""Same native ATCNet student for Q1/Q2; teacher is training-only in Q2."""
from __future__ import annotations

import hashlib
import random
import numpy as np
import tensorflow as tf

from published_covariance.atcnet_native import ATCNet_
from published_covariance.classifier import FrozenClassifier
from .protocol import initialization_seed


def state_digest(model):
    digest = hashlib.sha256()
    for value in model.weights:
        array = value.numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class Student:
    """Classify observed signals directly, without reconstructing hidden signals.

    Each student always observes exactly its target channel count. The 22 slots
    retain electrode identities for varying subsets; unavailable slots are zero.
    This is not a compact fixed-montage K-channel network or a completion model.
    """

    def __init__(self, subject, seed, retained):
        self.retained = retained
        init_seed = initialization_seed(subject, seed, retained)
        random.seed(init_seed)
        np.random.seed(init_seed)
        tf.random.set_seed(init_seed)
        self.model = ATCNet_(n_classes=4)
        self.model.trainable = True
        # Train from averaged branch logits, before the existing final softmax.
        self.logit_model = tf.keras.Model(self.model.inputs, self.model.get_layer("softmax").input)
        if self.model.count_params() != 115172 or not self.model.trainable_variables:
            raise RuntimeError("Unexpected student graph or frozen student")
        self.initial_state_sha256 = state_digest(self.model)

    def logits(self, values, mask, *, training):
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.bool)
        tf.debugging.assert_equal(tf.shape(values)[1:], [22, 1125])
        tf.debugging.assert_equal(tf.shape(mask), tf.shape(values)[:2])
        tf.debugging.assert_equal(tf.reduce_sum(tf.cast(mask, tf.int32), axis=1), self.retained)
        safe = tf.where(mask[..., None], values, tf.zeros_like(values))
        tf.debugging.assert_all_finite(safe, "Observed student input is nonfinite")
        logits = self.logit_model(safe[:, None, :, :], training=training)
        tf.debugging.assert_all_finite(logits, "Student logits are nonfinite")
        return logits

    def regularization(self):
        return tf.add_n(self.logit_model.losses) if self.logit_model.losses else tf.constant(0., tf.float32)


class Teacher:
    def __init__(self, subject, assets):
        self.classifier = FrozenClassifier(subject, assets_root=assets)
        self.model = self.classifier.model
        self.logit_model = tf.keras.Model(self.model.inputs, self.model.get_layer("softmax").input)
        self.logit_model.trainable = False
        self.state_before = state_digest(self.model)

    def logits(self, full_values):
        values = tf.convert_to_tensor(full_values, dtype=tf.float32)
        result = self.logit_model(values[:, None, :, :], training=False)
        tf.debugging.assert_all_finite(result, "Teacher logits are nonfinite")
        return tf.stop_gradient(result)

    def verify_frozen(self):
        if self.model.trainable_variables or state_digest(self.model) != self.state_before:
            raise RuntimeError("Full-channel teacher state changed")


def losses(student, logits, labels, teacher_logits, config):
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    kd = tf.constant(0., tf.float32)
    if config["distillation"]["enabled"]:
        if teacher_logits is None:
            raise ValueError("Q2 requires train-only teacher logits")
        temperature = config["distillation"]["temperature"]
        teacher_log_p = tf.nn.log_softmax(tf.stop_gradient(teacher_logits) / temperature)
        student_log_p = tf.nn.log_softmax(logits / temperature)
        kd = temperature ** 2 * tf.reduce_mean(tf.reduce_sum(
            tf.exp(teacher_log_p) * (teacher_log_p - student_log_p), axis=1))
    elif teacher_logits is not None:
        raise ValueError("Q1 must never consume teacher targets")
    regularization = student.regularization()
    total = ce + config["distillation"]["weight"] * kd + regularization
    tf.debugging.assert_all_finite(total, "Student training loss is nonfinite")
    return total, ce, kd, regularization
