"""Frozen local probes and paired raw-input robustness diagnostics."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from inm.availability import CHANNEL_IDS, make_mask_bank, mask_bank_digest
from .models import LOCAL_ARMS
from .training import _metrics


def _local_features(model, raw: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Extract frozen full-input electrode/window features without using labels."""
    if not hasattr(model, "freeze_features") or not hasattr(model, "forward_features"):
        raise TypeError("Local probes require a local classifier with a frozen feature encoder")
    model.freeze_features()
    model.eval()
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Local feature parameters remained trainable after freezing")
    outputs = []
    with torch.no_grad():
        for start in range(0, len(raw), batch_size):
            xb = torch.as_tensor(raw[start:start + batch_size], dtype=torch.float32,
                                 device=device)
            mask = torch.ones(xb.shape[:3], dtype=torch.bool, device=device)
            features = model.forward_features(xb, mask)
            if tuple(features.shape[1:]) != (22, 4, 32) or not torch.isfinite(features).all():
                raise FloatingPointError("Frozen local features must be finite [N,22,4,32]")
            outputs.append(features.detach().cpu().numpy())
    if not outputs:
        raise ValueError("Cannot extract features from an empty partition")
    return np.concatenate(outputs, axis=0)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _probe_arrays(features_train, features_validation, train_labels, validation_labels,
                  train_ids, validation_ids, seed, shuffled):
    """Fit fixed scaler/logistic probe and verify a numeric-array reload."""
    x_train = np.asarray(features_train, dtype=np.float64).reshape(len(train_labels), -1)
    x_validation = np.asarray(features_validation, dtype=np.float64).reshape(len(validation_labels), -1)
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise FloatingPointError("Probe features must be finite")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    used_labels = np.asarray(train_labels, dtype=np.int64).copy()
    if shuffled:
        np.random.default_rng(seed + 700001).shuffle(used_labels)
    classifier = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                                    tol=1e-4, random_state=0, class_weight=None)
    classifier.fit(x_train_scaled, used_labels)
    train_probabilities = classifier.predict_proba(x_train_scaled)
    validation_probabilities = classifier.predict_proba(x_validation_scaled)
    classes = np.asarray(classifier.classes_, dtype=np.int64)
    # Recreate the exact saved pipeline from numeric state, without pickle or
    # sklearn estimator serialization.
    reload_scaler = StandardScaler()
    reload_scaler.mean_ = scaler.mean_.copy()
    reload_scaler.scale_ = scaler.scale_.copy()
    reload_scaler.var_ = scaler.var_.copy()
    reload_scaler.n_features_in_ = scaler.n_features_in_
    reload_scaler.n_samples_seen_ = scaler.n_samples_seen_
    reload_classifier = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
        max_iter=1000, tol=1e-4, random_state=0, class_weight=None)
    reload_classifier.classes_ = classes.copy()
    reload_classifier.coef_ = classifier.coef_.copy()
    reload_classifier.intercept_ = classifier.intercept_.copy()
    reload_classifier.n_features_in_ = classifier.n_features_in_
    reloaded_train = reload_classifier.predict_proba(reload_scaler.transform(x_train))
    reloaded_validation = reload_classifier.predict_proba(reload_scaler.transform(x_validation))
    if (not np.allclose(train_probabilities, reloaded_train, rtol=1e-4, atol=1e-5)
            or not np.allclose(validation_probabilities, reloaded_validation,
                               rtol=1e-4, atol=1e-5)):
        raise RuntimeError("Numeric probe reload changed probabilities")
    converged = bool(classifier.n_iter_.max() < classifier.max_iter)
    return {
        "converged": converged,
        "train_metrics": _metrics(train_labels, train_probabilities),
        "validation_metrics": _metrics(validation_labels, validation_probabilities),
        "arrays": {
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "scaler_var": scaler.var_, "coef": classifier.coef_,
            "intercept": classifier.intercept_, "classes": classes,
            "train_sample_ids": np.asarray(train_ids, dtype=np.str_),
            "validation_sample_ids": np.asarray(validation_ids, dtype=np.str_),
            "train_labels_true": np.asarray(train_labels, dtype=np.int64),
            "train_labels_used": used_labels,
            "validation_labels_true": np.asarray(validation_labels, dtype=np.int64),
            "train_probabilities": train_probabilities,
            "validation_probabilities": validation_probabilities,
            "train_reloaded_probabilities": reloaded_train,
            "validation_reloaded_probabilities": reloaded_validation,
            "n_iter": np.asarray(classifier.n_iter_, dtype=np.int64),
            "converged": np.asarray(classifier.n_iter_.max() < classifier.max_iter,
                                     dtype=np.bool_),
            "random_state": np.asarray(0, dtype=np.int64),
            "shuffle_seed": np.asarray(seed + 700001 if shuffled else -1, dtype=np.int64),
        },
    }


def _verify_saved_probe(path: Path, features_train: np.ndarray,
                       features_validation: np.ndarray, expected: dict) -> None:
    """Reload the on-disk NPZ with pickle disabled and reproduce probabilities."""
    train = np.asarray(features_train, dtype=np.float64).reshape(len(expected["arrays"]["train_labels_true"]), -1)
    validation = np.asarray(features_validation, dtype=np.float64).reshape(
        len(expected["arrays"]["validation_labels_true"]), -1)
    with np.load(path, allow_pickle=False) as saved:
        scaler = StandardScaler()
        scaler.mean_ = saved["scaler_mean"].copy()
        scaler.scale_ = saved["scaler_scale"].copy()
        scaler.var_ = saved["scaler_var"].copy()
        scaler.n_features_in_ = int(train.shape[1])
        scaler.n_samples_seen_ = int(len(train))
        classifier = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                        max_iter=1000, tol=1e-4,
                                        random_state=0, class_weight=None)
        classifier.classes_ = saved["classes"].copy()
        classifier.coef_ = saved["coef"].copy()
        classifier.intercept_ = saved["intercept"].copy()
        classifier.n_features_in_ = int(train.shape[1])
        actual_train = classifier.predict_proba(scaler.transform(train))
        actual_validation = classifier.predict_proba(scaler.transform(validation))
        if (not np.allclose(actual_train, saved["train_probabilities"], rtol=1e-4, atol=1e-5)
                or not np.allclose(actual_validation, saved["validation_probabilities"],
                                   rtol=1e-4, atol=1e-5)
                or not np.allclose(actual_train, expected["arrays"]["train_probabilities"],
                                   rtol=1e-4, atol=1e-5)
                or not np.allclose(actual_validation,
                                   expected["arrays"]["validation_probabilities"],
                                   rtol=1e-4, atol=1e-5)):
            raise RuntimeError("Reloading the saved numeric probe changed probabilities")


def fit_local_probes(model, prepared, cfg, directory):
    """Fit ordered and shuffled-label probes for one selected local model.

    The model itself is retained for future audits; only its local encoder is
    frozen while train and validation features are extracted. Probe artifacts
    are numeric NPZ files and can be read with ``allow_pickle=False``.
    """
    arm = getattr(model, "kind", None)
    if arm not in LOCAL_ARMS:
        raise ValueError("fit_local_probes accepts only local_control/local_power")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be the loaded candidate configuration")
    settings = cfg.get("probes", {})
    if settings and (settings.get("C") != 1.0 or settings.get("max_iter") != 1000
                     or settings.get("solver") != "lbfgs" or settings.get("tol") != 1e-4):
        raise ValueError("Probe settings differ from the fixed candidate protocol")
    train_y = np.asarray(prepared.train.labels, dtype=np.int64)
    val_y = np.asarray(prepared.validation.labels, dtype=np.int64)
    if len(train_y) != len(prepared.train.sample_ids) or len(val_y) != len(prepared.validation.sample_ids):
        raise ValueError("Probe labels and stable sample IDs have inconsistent lengths")
    device = next(model.parameters()).device
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("Probe extraction batch_size must be positive")
    train_features = _local_features(model, prepared.train.raw, batch_size, device)
    validation_features = _local_features(model, prepared.validation.raw, batch_size, device)
    output = Path(directory)
    results = {}
    for name, shuffled in (("probe", False), ("probe_shuffled", True)):
        result = _probe_arrays(train_features, validation_features, train_y, val_y,
                               prepared.train.sample_ids, prepared.validation.sample_ids,
                               int(prepared.seed), shuffled)
        path = output / f"{name}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing probe artifact {path}")
        _atomic_npz(path, **result["arrays"])
        _verify_saved_probe(path, train_features, validation_features, result)
        result.pop("arrays")
        result["path"] = str(path)
        results[name] = result
    return results


def _mask_model(model, raw: np.ndarray, masks: np.ndarray, batch_size: int,
                device: torch.device) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(raw), batch_size):
            xb = torch.as_tensor(raw[start:start + batch_size], dtype=torch.float32,
                                 device=device)
            mb = torch.as_tensor(masks[start:start + batch_size], dtype=torch.bool,
                                 device=device)
            if not mb.any(dim=1).all():
                raise ValueError("Every evaluated window must retain an observed electrode")
            # Pass original raw values and the original-ID mask to the model.
            # Models must select hidden values before filtering/mixing.
            logits = model(xb, mb)
            if tuple(logits.shape) != (len(xb), 4) or not torch.isfinite(logits).all():
                raise FloatingPointError("Masked raw-input evaluation produced invalid logits")
            probabilities.append(logits.softmax(dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def evaluate_selected(model, prepared, cfg):
    """Score the selected model on paired validation masks from raw signals.

    Returns 21 rows (one full-input row and five repeats for each degraded
    condition). It never reads or creates a test partition and never applies a
    loss mask to spatial activations or cached features.
    """
    validation = prepared.validation
    raw = np.asarray(validation.raw)
    labels = np.asarray(validation.labels, dtype=np.int64)
    ids = tuple(validation.sample_ids)
    if raw.ndim != 4 or raw.shape[1:] != (22, 4, 250) or len(raw) != len(labels):
        raise ValueError("Validation raw data must be [N,22,4,250] with matching labels")
    if len(ids) != len(raw) or len(ids) != len(set(ids)):
        raise ValueError("Validation sample IDs must be unique and aligned with raw inputs")
    if not np.isfinite(raw).all():
        raise ValueError("Validation raw EEG contains non-finite values")
    conditions = cfg.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 5:
        raise ValueError("Candidate config must define the five fixed robustness conditions")
    expected = (("full_22", "full", 22, 1),
                ("random_static_16", "random_static", 16, 5),
                ("dynamic_random_16", "dynamic_random", 16, 5),
                ("random_static_6", "random_static", 6, 5),
                ("dynamic_random_6", "dynamic_random", 6, 5))
    declared = tuple((row.get("name"), row.get("pattern"), row.get("retained"), row.get("repeats"))
                     for row in conditions)
    if declared != expected:
        raise ValueError("Robustness conditions differ from the fixed five-condition protocol")
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be positive")
    try:
        parameter = next(model.parameters())
        device = parameter.device
    except StopIteration:
        device = torch.device("cpu")
    subject = f"A{int(prepared.subject):02d}"
    rows = []
    for scenario, pattern, retained, repeats in expected:
        for repeat in range(repeats):
            masks = make_mask_bank(len(raw), 4, retained, pattern,
                seed=int(prepared.seed), partition="validation", subject=subject,
                repeat=repeat, sample_ids=ids, channel_ids=CHANNEL_IDS)
            if masks.shape != (len(raw), 22, 4) or masks.dtype != np.bool_:
                raise RuntimeError("Mask bank returned an invalid original-channel mask")
            if not masks.any(axis=1).all():
                raise ValueError("Mask bank contains an all-missing window")
            probabilities = _mask_model(model, raw, masks, batch_size, device)
            metrics = _metrics(labels, probabilities)
            rows.append({"scenario": scenario, "repeat": repeat,
                         "mask_sha256": mask_bank_digest(masks), **metrics})
    if len(rows) != 21 or not np.isfinite([row["balanced_accuracy"] for row in rows]).all():
        raise FloatingPointError("Robustness evaluation did not produce 21 finite validation rows")
    return rows
