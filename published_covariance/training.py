"""Covariance-only training for an audited frozen published classifier."""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
import tensorflow as tf

from .completion import CovarianceCompleter, hidden_mse
from .masks import static_masks


def select_checkpoint(records):
    """Select maximum degraded BA, then minimum log loss, then earliest epoch."""
    rows = list(records)
    if not rows:
        raise ValueError("at least one checkpoint record is required")
    for row in rows:
        if not isinstance(row, dict) or not {"epoch", "degraded_ba", "degraded_log_loss"} <= row.keys():
            raise ValueError("checkpoint records need epoch, degraded_ba, and degraded_log_loss")
        epoch = row["epoch"]
        if not isinstance(epoch, (int, np.integer)) or isinstance(epoch, (bool, np.bool_)) or epoch < 0:
            raise ValueError("checkpoint epochs must be nonnegative integers")
        if not math.isfinite(float(row["degraded_ba"])) or not math.isfinite(float(row["degraded_log_loss"])):
            raise ValueError("checkpoint scores must be finite")
    return min(rows, key=lambda row: (-float(row["degraded_ba"]),
                                      float(row["degraded_log_loss"]), int(row["epoch"])))


def _ids(values, name, count):
    result = tuple(str(value) for value in values)
    if len(result) != count or len(set(result)) != count or any(not x for x in result):
        raise ValueError(f"{name} must contain one unique nonempty ID per trial")
    return result


def _validate_t_ids(ids, subject, partition):
    pattern = re.compile(r"(A0[1-9]):T:r\d{2}:t\d{3}\Z")
    for trial_id in ids:
        match = pattern.fullmatch(trial_id)
        if match is None or match.group(1) != subject:
            raise ValueError(f"{partition} IDs must identify {subject} T-session trials")


def _probabilities(classifier, values):
    output = classifier(values, training=False)
    if isinstance(output, (tuple, list)):
        raise TypeError("classifier must return one probability tensor")
    probabilities = tf.convert_to_tensor(output)
    checks = [tf.debugging.assert_rank(probabilities, 2),
              tf.debugging.assert_equal(tf.shape(probabilities), [tf.shape(values)[0], 4]),
              tf.debugging.assert_all_finite(probabilities, "classifier probabilities are nonfinite"),
              tf.debugging.assert_greater_equal(probabilities, tf.cast(0., probabilities.dtype)),
              tf.debugging.assert_near(tf.reduce_sum(probabilities, axis=1),
                                       tf.ones([tf.shape(values)[0]], probabilities.dtype),
                                       atol=1e-4, rtol=1e-4)]
    with tf.control_dependencies(checks):
        return tf.identity(probabilities)


def _scores(probabilities, labels):
    pred = np.argmax(probabilities, axis=1)
    recalls = [np.mean(pred[labels == c] == c) for c in range(4) if np.any(labels == c)]
    ba = float(np.mean(recalls))
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return ba, float(-np.mean(np.log(chosen)))


def _validation_record(classifier, completer, values, labels, ids, subject, seed, epoch):
    bas, losses = [], []
    for retained in (16, 6):
        for repeat in range(5):
            mask = static_masks(subject, seed, ids, "validation", retained,
                                repeat=repeat, epoch=0)
            completed = completer.complete(values, mask)
            probabilities = _probabilities(classifier, completed).numpy()
            ba, loss = _scores(probabilities, labels)
            bas.append(ba)
            losses.append(loss)
    return {"epoch": int(epoch), "degraded_ba": float(np.mean(bas)),
            "degraded_log_loss": float(np.mean(losses)),
            "validation_ba_by_condition": bas,
            "validation_log_loss_by_condition": losses}


def train_covariance(classifier, train_values, train_labels, train_ids,
                     validation_values, validation_labels, validation_ids,
                     subject, seed, max_epochs=100, min_epochs=10, patience=20,
                     synthetic=False) -> dict:
    """Fit covariance only, using T-train for initialization and T-validation for selection.

    ``synthetic=True`` is an explicit test hook permitting reduced epoch/minimum/
    patience budgets. It does not change the objective, optimizer or mask protocol.
    """
    if not isinstance(synthetic, (bool, np.bool_)):
        raise TypeError("synthetic must be Boolean")
    if not isinstance(subject, str) or subject not in {f"A{i:02d}" for i in range(1, 10)}:
        raise ValueError("subject must be A01 through A09")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (("max_epochs", max_epochs), ("min_epochs", min_epochs), ("patience", patience)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")
    if max_epochs < 1 or min_epochs < 0 or min_epochs > max_epochs or patience < 1:
        raise ValueError("invalid epoch or patience budget")
    if not synthetic and (max_epochs != 100 or min_epochs != 10 or patience != 20):
        raise ValueError("real training uses the fixed 100/10/20 budget")

    train = np.asarray(train_values)
    valid = np.asarray(validation_values)
    y_train = np.asarray(train_labels)
    y_valid = np.asarray(validation_labels)
    for name, values, labels in (("training", train, y_train), ("validation", valid, y_valid)):
        if values.ndim != 3 or values.shape[1] != 22 or values.shape[2] < 1 or values.shape[0] < 1:
            raise ValueError(f"{name} values must have shape [N,22,T]")
        if values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"{name} values must be finite floating-point data")
        if labels.shape != (len(values),) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels > 3)):
            raise ValueError(f"{name} labels must be integer class IDs 0..3")
    train_id = _ids(train_ids, "train_ids", len(train))
    validation_id = _ids(validation_ids, "validation_ids", len(valid))
    _validate_t_ids(train_id, subject, "training")
    _validate_t_ids(validation_id, subject, "validation")
    if set(train_id) & set(validation_id):
        raise ValueError("training and validation trial IDs must be disjoint")
    if not callable(classifier):
        raise TypeError("classifier must be callable")

    # Uncentered training-only second moment over every trial and time sample.
    train64 = train.astype(np.float64, copy=False)
    second_moment = np.einsum("nct,ndt->cd", train64, train64, optimize=True)
    second_moment /= float(len(train) * train.shape[2])
    completer = CovarianceCompleter(second_moment, trainable=True)
    initial = completer.free.numpy().copy()

    # Stable local stream; no process-global Python, NumPy or TensorFlow RNG calls.
    seed_payload = json.dumps(["published-covariance-train-v1", subject, int(seed)],
                              separators=(",", ":")).encode()
    local_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:16], "little")
    rng = np.random.Generator(np.random.PCG64(local_seed))
    train_tf = tf.convert_to_tensor(train)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)
    valid_tf = tf.convert_to_tensor(valid)

    # Ensure classifier parameters are frozen when a Keras model is exposed.
    model = getattr(classifier, "model", None)
    if model is not None and hasattr(model, "trainable"):
        model.trainable = False
        if getattr(model, "trainable_variables", ()):
            raise RuntimeError("classifier model still has trainable variables")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                         beta_2=0.999, epsilon=1e-8)
    history = []
    initial_record = _validation_record(classifier, completer, valid_tf, y_valid,
                                        validation_id, subject, int(seed), 0)
    initial_record.update({"training_objective": None, "cross_entropy": None,
                           "hidden_mse": None, "regularization": None,
                           "parameter_delta_norm": 0.0, "gradient_norm_mean": None})
    history.append(initial_record)
    best_record = select_checkpoint(history)
    selected_free = initial.copy()
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(train))
        epoch_losses, epoch_ce, epoch_hidden, epoch_reg, grad_norms = [], [], [], [], []
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            counts = rng.choice(np.array([22, 16, 6], dtype=np.int32), size=len(indices))
            masks = np.ones((len(indices), 22), dtype=np.bool_)
            for retained in (16, 6):
                positions = np.flatnonzero(counts == retained)
                if len(positions):
                    masks[positions] = static_masks(
                        subject, int(seed), [train_id[int(i)] for i in indices[positions]],
                        "train", retained, repeat=0, epoch=epoch)
            x = tf.gather(train_tf, indices)
            mask_tf = tf.convert_to_tensor(masks)
            labels = tf.gather(y_train_tf, indices)
            with tf.GradientTape() as tape:
                completed = completer.complete(x, mask_tf)
                probabilities = _probabilities(classifier, completed)
                ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    labels, probabilities, from_logits=False))
                reconstruction = hidden_mse(completed, x, mask_tf)
                regularization = 1e-4 * tf.reduce_mean(tf.square(completer.free - completer.initial_free))
                objective = tf.cast(ce, tf.float64) + 0.1 * reconstruction + regularization
            gradient = tape.gradient(objective, completer.free)
            if gradient is None:
                raise RuntimeError("covariance objective has no gradient")
            tf.debugging.assert_all_finite(gradient, "covariance gradient is nonfinite")
            clipped, grad_norm = tf.clip_by_global_norm([gradient], 1.0)
            optimizer.apply_gradients([(clipped[0], completer.free)])
            epoch_losses.append(float(objective.numpy()))
            epoch_ce.append(float(ce.numpy()))
            epoch_hidden.append(float(reconstruction.numpy()))
            epoch_reg.append(float(regularization.numpy()))
            grad_norms.append(float(grad_norm.numpy()))

        record = _validation_record(classifier, completer, valid_tf, y_valid,
                                    validation_id, subject, int(seed), epoch)
        record.update({"training_objective": float(np.mean(epoch_losses)),
                       "cross_entropy": float(np.mean(epoch_ce)),
                       "hidden_mse": float(np.mean(epoch_hidden)),
                       "regularization": float(np.mean(epoch_reg)),
                       "parameter_delta_norm": float(np.linalg.norm(completer.free.numpy() - initial)),
                       "gradient_norm_mean": float(np.mean(grad_norms))})
        history.append(record)
        chosen = select_checkpoint(history)
        improved = (int(chosen["epoch"]) == epoch)
        if improved:
            best_record = record
            selected_free = completer.free.numpy().copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    completer.free.assign(selected_free)
    return {"completer": completer, "history": history,
            "selected_epoch": int(best_epoch), "selected_score": {
                "degraded_ba": float(best_record["degraded_ba"]),
                "degraded_log_loss": float(best_record["degraded_log_loss"])},
            "selected_record": dict(best_record), "selected_free": selected_free.copy(),
            "initial_free": initial.copy(), "epochs_completed": len(history) - 1,
            "stopped_early": len(history) - 1 < max_epochs,
            "patience": int(patience), "min_epochs": int(min_epochs),
            "max_epochs": int(max_epochs)}
