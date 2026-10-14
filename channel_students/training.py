"""Matched student fitting: Q1 is CE; Q2 adds temperature-scaled teacher KL."""
from __future__ import annotations

import json
import time
import numpy as np
import tensorflow as tf

from published_covariance.masks import static_masks
from .artifacts import write_json
from .model import losses
from .protocol import order_seed


def metrics(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if (p.shape != (len(y), 4) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid four-class probabilities")
    if not len(y) or set(y.tolist()) != {0, 1, 2, 3}:
        raise ValueError("Every declared partition must retain all four classes")
    predicted = p.argmax(axis=1)
    return {"balanced_accuracy": float(np.mean([np.mean(predicted[y == c] == c) for c in range(4)])),
            "accuracy": float(np.mean(predicted == y)),
            "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))),
            "n_trials": int(len(y))}


def predict(student, values, mask, batch_size):
    probabilities = []
    for start in range(0, len(values), batch_size):
        logits = student.logits(values[start:start + batch_size], mask[start:start + batch_size], training=False)
        probabilities.append(tf.nn.softmax(logits).numpy())
    return np.concatenate(probabilities)


def selection(history):
    if not history:
        raise ValueError("Checkpoint history is empty")
    return min(history, key=lambda row: (-row["validation_ba"], row["validation_log_loss"], row["epoch"]))


def validation(student, x, y, masks, batch_size, epoch):
    probabilities = np.stack([predict(student, x, mask, batch_size) for mask in masks])
    scores = [metrics(p, y) for p in probabilities]
    return {"epoch": epoch,
            "validation_ba": float(np.mean([m["balanced_accuracy"] for m in scores])),
            "validation_log_loss": float(np.mean([m["log_loss"] for m in scores]))}, probabilities


def make_train_step(student, config):
    """One traced production step, also exercised by cluster GPU admission."""
    options = config["training"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=options["learning_rate"],
                                         beta_1=options["betas"][0], beta_2=options["betas"][1],
                                         epsilon=options["epsilon"])

    @tf.function(input_signature=[tf.TensorSpec([None, 22, 1125], tf.float32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None, 22], tf.bool),
                                  tf.TensorSpec([None, 4], tf.float32)])
    def step(values, labels, mask, targets):
        with tf.GradientTape() as tape:
            logits = student.logits(values, mask, training=True)
            objective, ce, kd, regularization = losses(
                student, logits, labels,
                targets if config["distillation"]["enabled"] else None, config)
        gradients = tape.gradient(objective, student.logit_model.trainable_variables)
        for gradient in gradients:
            if gradient is None:
                raise RuntimeError("A student parameter has no training gradient")
            tf.debugging.assert_all_finite(gradient, "Student gradient is nonfinite")
        clipped, norm = tf.clip_by_global_norm(gradients, options["gradient_clip_global_norm"])
        optimizer.apply_gradients(zip(clipped, student.logit_model.trainable_variables))
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels), tf.int32))
        return objective, ce, kd, regularization, norm, correct
    return step


def fit(student, train_x, train_y, train_ids, validation_x, validation_y,
        validation_ids, subject, seed, config, folder, teacher_logits=None):
    options = config["training"]
    retained = student.retained
    if config["distillation"]["enabled"] != (teacher_logits is not None):
        raise ValueError("Teacher targets must be present exclusively in Q2")
    if teacher_logits is not None and (teacher_logits.shape != (len(train_x), 4)
                                     or not np.isfinite(teacher_logits).all()):
        raise ValueError("Teacher targets do not match T training trials")
    valid_masks = np.stack([static_masks(
        subject, seed, validation_ids, "validation", retained, repeat=r,
        namespace=config["mask_namespace"]) for r in range(config["mask_repeats"])])
    step = make_train_step(student, config)

    rng = np.random.Generator(np.random.PCG64(order_seed(subject, seed, retained)))
    initial_record, selected_probabilities = validation(
        student, validation_x, validation_y, valid_masks, options["batch_size"], 0)
    history = [initial_record]
    selected_weights = student.model.get_weights()
    best_epoch = 0
    stale_epochs = 0
    orders, train_masks = [], []
    started = time.perf_counter()
    for epoch in range(1, options["max_epochs"] + 1):
        order = rng.permutation(len(train_x))
        mask = static_masks(subject, seed, train_ids, "train", retained, epoch=epoch,
                            namespace=config["mask_namespace"])
        orders.append(order)
        train_masks.append(mask)
        sums = np.zeros(5, dtype=np.float64)
        correct = 0
        for start in range(0, len(order), options["batch_size"]):
            indices = order[start:start + options["batch_size"]]
            targets = (teacher_logits[indices] if teacher_logits is not None
                       else np.zeros((len(indices), 4), np.float32))
            result = step(tf.convert_to_tensor(train_x[indices]),
                          tf.convert_to_tensor(train_y[indices], dtype=tf.int32),
                          tf.convert_to_tensor(mask[indices]), tf.convert_to_tensor(targets))
            sums += np.asarray([float(v.numpy()) for v in result[:5]]) * len(indices)
            correct += int(result[5].numpy())
        record, probabilities = validation(
            student, validation_x, validation_y, valid_masks, options["batch_size"], epoch)
        record.update(dict(zip(("objective", "cross_entropy", "distillation_kl", "regularization", "gradient_norm"),
                               (sums / len(train_x)).tolist())))
        record["training_accuracy"] = correct / len(train_x)
        history.append(record)
        if selection(history)["epoch"] == epoch:
            selected_weights = student.model.get_weights()
            selected_probabilities = probabilities
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        write_json(folder / "history.json", history, replace=True)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"experiment": config["experiment"], "subject": subject,
                              "seed": seed, "retained": retained, **record}), flush=True)
        if epoch >= options["min_epochs"] and stale_epochs >= options["patience"]:
            break
    student.model.set_weights(selected_weights)
    return {"history": history, "selected_epoch": best_epoch,
            "selected_validation_probabilities": selected_probabilities,
            "validation_masks": valid_masks, "train_masks": np.asarray(train_masks),
            "epoch_orders": np.asarray(orders), "training_seconds": time.perf_counter() - started}
