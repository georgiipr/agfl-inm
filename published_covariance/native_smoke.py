"""Generated-input native-checkpoint CUDA integration smoke.

This path is intentionally separate from real study outputs and never loads EEG
recordings or labels. It exercises the admitted checkpoints and a tiny
covariance-only fit before the supervisor starts real fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from .classifier import FrozenClassifier
from .training import train_covariance


def _state_hash(models):
    digest = hashlib.sha256()
    for subject in sorted(models):
        model = models[subject].model
        for variable in model.weights:
            value = np.asarray(variable.numpy())
            digest.update(subject.encode())
            digest.update(variable.name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def run_native_smoke(output: str | Path, assets_root: str | Path, device="cuda:0"):
    """Run all-nine native GPU gradients and an A01 one-epoch synthetic fit."""
    if device != "cuda:0":
        raise RuntimeError("native synthetic smoke requires cuda:0; CPU fallback is forbidden")
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("native synthetic smoke requires a TensorFlow-visible GPU")
    # The public CLI sets memory growth before importing any model. Reassert it
    # here and fail if this helper was called after GPU initialization.
    for gpu in physical:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError("GPU memory growth must be configured before model initialization") from exc

    target = Path(output).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("native smoke output path cannot contain symlinks")
    target = target.resolve()
    results = (Path(__file__).resolve().parents[1] / "results").resolve()
    if target == results or results in target.parents:
        raise ValueError("native smoke output must stay outside real results")
    target.mkdir(parents=True, exist_ok=False)

    rng = np.random.Generator(np.random.PCG64(20261006))
    models = {}
    before = None
    started = time.monotonic()
    checked = []
    for index in range(1, 10):
        subject = f"A{index:02d}"
        with tf.device("/GPU:0"):
            model = FrozenClassifier(subject, assets_root=assets_root)
            values = rng.normal(size=(1, 22, 1125)).astype(np.float32)
            x = tf.Variable(values)
            with tf.GradientTape() as tape:
                probabilities = model(x)
                scalar = tf.reduce_sum(probabilities[:, 0])
            gradient = tape.gradient(scalar, x)
        if gradient is None:
            raise RuntimeError(f"{subject} native input gradient is missing")
        if "GPU:0" not in probabilities.device or "GPU:0" not in gradient.device:
            raise RuntimeError(f"{subject} native forward/backward did not execute on GPU")
        if not np.isfinite(gradient.numpy()).all() or not np.linalg.norm(gradient.numpy()) > 0:
            raise RuntimeError(f"{subject} native input gradient is not finite and nonzero")
        if not np.isfinite(probabilities.numpy()).all() or model.model.trainable_variables:
            raise RuntimeError(f"{subject} native forward or frozen-state check failed")
        models[subject] = model
        checked.append(subject)
    before = _state_hash(models)

    # Small, generated T-session arrays: four disjoint training IDs and four
    # validation IDs, with one actual update and no recording/data adapter.
    train_values = rng.normal(size=(4, 22, 1125)).astype(np.float32)
    validation_values = rng.normal(size=(4, 22, 1125)).astype(np.float32)
    labels = np.arange(4, dtype=np.int32)
    train_ids = [f"A01:T:r00:t{i:03d}" for i in range(4)]
    validation_ids = [f"A01:T:r00:t{i:03d}" for i in range(4, 8)]
    with tf.device("/GPU:0"):
        result = train_covariance(
            models["A01"], train_values, labels, train_ids,
            validation_values, labels, validation_ids,
            "A01", seed=0, max_epochs=1, min_epochs=0, patience=1,
            synthetic=True)

        # This saved probe has missing channels, so the public checker exercises
        # conditional completion and native classifier replay on actual GPU.
        masks = np.ones((4, 22), dtype=np.bool_)
        masks[:, 16:] = False
        probe = tf.Variable(validation_values)
        probe_mask = tf.convert_to_tensor(masks)
        with tf.GradientTape() as tape:
            completed = result["completer"].complete(probe, probe_mask)
            probe_probs = models["A01"](completed)
            probe_loss = tf.reduce_sum(probe_probs[:, 0])
        covariance_gradient, input_gradient = tape.gradient(
            probe_loss, [result["completer"].free, probe])

    if "GPU:0" not in probe_probs.device or "GPU:0" not in completed.device:
        raise RuntimeError("synthetic covariance completion/classifier did not execute on GPU")
    for name, gradient in (("covariance", covariance_gradient), ("input", input_gradient)):
        if gradient is None or not np.isfinite(gradient.numpy()).all() or not np.linalg.norm(gradient.numpy()) > 0:
            raise RuntimeError(f"synthetic {name} gradient is not finite and nonzero")
    if len(result["history"]) < 2 or not any(
            row.get("parameter_delta_norm", 0.0) > 0 for row in result["history"][1:]):
        raise RuntimeError("synthetic covariance fit did not perform an optimizer update")

    # Restore precisely the selected state, then prove saved/reloaded inference
    # by assigning it into a fresh covariance completer before serialization.
    from .completion import CovarianceCompleter
    selected_path = target / "selected-covariance-free.npy"
    with selected_path.open("xb") as stream:
        np.save(stream, result["selected_free"], allow_pickle=False)
    selected_free_reloaded = np.load(selected_path, allow_pickle=False)
    if not np.array_equal(selected_free_reloaded, result["selected_free"]):
        raise RuntimeError("saved selected covariance state did not reload exactly")
    replay_completer = CovarianceCompleter(np.eye(22, dtype=np.float64), trainable=False)
    replay_completer.free.assign(selected_free_reloaded)
    with tf.device("/GPU:0"):
        selected_completed = replay_completer.complete(validation_values, masks)
        probabilities = models["A01"](selected_completed).numpy()
        replay_completer.free.assign(np.load(selected_path, allow_pickle=False))
        replay_completed = replay_completer.complete(validation_values, masks)
        replay_probabilities = models["A01"](replay_completed).numpy()
    if not np.array_equal(probabilities, replay_probabilities):
        raise RuntimeError("selected covariance state replay changed native probabilities")

    after = _state_hash(models)
    if before != after:
        raise RuntimeError("native checkpoint parameters or BatchNorm state changed during fit")
    np.savez(
        target / "native-smoke.npz",
        values=validation_values,
        mask=masks,
        initial_free=result["initial_free"],
        selected_free=result["selected_free"],
        probabilities=probabilities,
        replay_probabilities=replay_probabilities,
        covariance_gradient=covariance_gradient.numpy(),
        input_gradient=input_gradient.numpy(),
    )
    metadata = {
        "schema": "published-covariance-native-smoke-v1",
        "synthetic_fixture": True,
        "device": "cuda:0",
        "subjects": checked,
        "frozen_before_sha256": before,
        "frozen_after_sha256": after,
        "history": result["history"],
        "selected_epoch": int(result["selected_epoch"]),
        "generated_training_ids": train_ids,
        "generated_validation_ids": validation_ids,
        "training_max_epochs": 1,
        "training_updates": len(result["history"]) - 1,
        "output_files": ["native-smoke.json", "native-smoke.npz"],
        "outcome_scoring": False,
        "real_fits": 0,
        "real_E_predictions": 0,
        "seconds": time.monotonic() - started,
    }
    with (target / "native-smoke.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata
