"""Fixed, training-fitted linear probes for legacy encoder representations.

Only the verified legacy train/validation views are accepted. Probe files use
numeric arrays plus JSON metadata embedded as a Unicode scalar; loading never
uses pickle.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import warnings


class ProbeError(ValueError):
    """A probe input, fit, or persisted probe is invalid."""


VIEWS = {
    "features_ordered": {"shape": [22, 4, 32], "layout": "C-order flatten [22,4,32]", "width": 2816},
    "features_window_mean": {"shape": [22, 32], "layout": "mean axis=2 then C-order flatten [22,32]", "width": 704},
    "core_ordered": {"shape": [4, 4, 4], "layout": "C-order flatten [4,4,4] from [rank_channel,window,rank_feature]", "width": 64},
    "features_ordered_shuffled": {"shape": [22, 4, 32], "layout": "C-order flatten [22,4,32]; shuffled training labels", "width": 2816},
}
CLASSES = (0, 1, 2, 3)


def _finite_features(value, name, shape):
    import numpy as np
    array = np.asarray(value)
    if array.ndim != 4 or tuple(array.shape[1:]) != tuple(shape):
        raise ProbeError(f"{name} must have shape [N,{','.join(map(str, shape))}], got {array.shape}")
    if len(array) == 0 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ProbeError(f"{name} must be nonempty and finite")
    return array.astype(np.float64, copy=False)


def _labels(value, count, name):
    import numpy as np
    labels = np.asarray(value)
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ProbeError(f"{name} labels must be an integer vector of length {count}")
    if not np.isin(labels, CLASSES).all():
        raise ProbeError(f"{name} labels must be in {CLASSES}")
    if set(labels.tolist()) != set(CLASSES):
        raise ProbeError(f"{name} must contain all four classes")
    return labels.astype(np.int64, copy=False)


def _views(sources):
    import numpy as np
    legacy = sources.legacy
    train = _finite_features(legacy.train_features, "train features", (22, 4, 32))
    validation = _finite_features(legacy.validation_features, "validation features", (22, 4, 32))
    y_train = _labels(legacy.train.labels, len(train), "train")
    y_validation = _labels(legacy.validation.labels, len(validation), "validation")
    if len(legacy.train.sample_ids) != len(train) or len(set(legacy.train.sample_ids)) != len(train):
        raise ProbeError("Training sample IDs are missing, duplicated, or misaligned")
    if len(legacy.validation.sample_ids) != len(validation) or len(set(legacy.validation.sample_ids)) != len(validation):
        raise ProbeError("Validation sample IDs are missing, duplicated, or misaligned")
    if set(legacy.train.sample_ids) & set(legacy.validation.sample_ids):
        raise ProbeError("Training and validation sample IDs overlap")
    u = legacy.factor_state.get("U")
    v = legacy.factor_state.get("V")
    if u is None or v is None:
        raise ProbeError("Saved Tucker factors must contain U and V")
    u_array = u.detach().cpu().numpy() if hasattr(u, "detach") else np.asarray(u)
    v_array = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    if u_array.ndim != 2 or u_array.shape[0] != 22 or v_array.ndim != 2 or v_array.shape[0] != 32:
        raise ProbeError("Saved Tucker factors have unsupported dimensions")
    if not np.isfinite(u_array).all() or not np.isfinite(v_array).all():
        raise ProbeError("Saved Tucker factors contain nonfinite values")
    return train, validation, y_train, y_validation, u_array.copy(), v_array.copy()


def _feature_views(train, validation, factor_state, factors):
    import numpy as np
    train_views = {
        "features_ordered": train.reshape(len(train), -1),
        "features_window_mean": train.mean(axis=2).reshape(len(train), -1),
    }
    validation_views = {
        "features_ordered": validation.reshape(len(validation), -1),
        "features_window_mean": validation.mean(axis=2).reshape(len(validation), -1),
    }
    # Loading and inference use the existing implementation; this fixed full
    # mask is independent of labels and does not update the saved buffers.
    import torch
    from inm.tensor_attention import Tucker2
    u, v = factors
    state = {}
    for key, value in factor_state.items():
        state[key] = value.detach().cpu().clone() if hasattr(value, "detach") else torch.as_tensor(value).clone()
    solver = Tucker2(channels=22, features=32, rank_channels=u.shape[1], rank_features=v.shape[1], ridge=0.001)
    try:
        solver.load_state_dict(state, strict=True)
    except Exception as error:
        raise ProbeError(f"Saved Tucker factor state cannot be loaded: {error}") from error
    solver.eval()
    if not bool(solver.fitted.item()):
        raise ProbeError("Saved Tucker factor state is not marked fitted")
    before = {key: value.detach().clone() for key, value in solver.state_dict().items()}
    with torch.inference_mode():
        train_core = solver.encode(torch.as_tensor(train, dtype=torch.float32),
            torch.ones((len(train), 22, 4), dtype=torch.bool)).cpu().numpy()
        validation_core = solver.encode(torch.as_tensor(validation, dtype=torch.float32),
            torch.ones((len(validation), 22, 4), dtype=torch.bool)).cpu().numpy()
    after = solver.state_dict()
    if any(not torch.equal(before[key], after[key]) for key in before):
        raise ProbeError("Tucker inference mutated the frozen saved factor state")
    train_views["core_ordered"] = train_core.reshape(len(train_core), -1)
    validation_views["core_ordered"] = validation_core.reshape(len(validation_core), -1)
    if any(not np.isfinite(x).all() for x in (*train_views.values(), *validation_views.values())):
        raise ProbeError("A derived probe view contains nonfinite values")
    return train_views, validation_views


class Probe:
    """Numeric scaler/classifier state with prediction from normalized features."""
    def __init__(self, name, mean, scale, coef, intercept, classes, metadata):
        self.name = name
        self.mean = mean
        self.scale = scale
        self.coef = coef
        self.intercept = intercept
        self.classes = classes
        self.metadata = metadata

    def _matrix(self, values):
        import numpy as np
        array = np.asarray(values)
        if self.name in ("features_ordered", "features_ordered_shuffled") and array.ndim == 4:
            if tuple(array.shape[1:]) != (22, 4, 32):
                raise ProbeError("Ordered-feature prediction expects normalized [N,22,4,32]")
            array = array.reshape(len(array), -1)
        elif array.ndim != 2:
            raise ProbeError("Probe prediction expects a 2D matrix (or ordered [N,22,4,32] tensor)")
        if array.shape[1] != len(self.mean) or not np.isfinite(array).all():
            raise ProbeError("Probe prediction features have wrong width or nonfinite values")
        return array.astype(np.float64, copy=False)

    def predict_proba(self, normalized_features):
        import numpy as np
        x = self._matrix(normalized_features)
        z = (x - self.mean) / self.scale
        logits = z @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, normalized_features):
        return self.classes[self.predict_proba(normalized_features).argmax(axis=1)]


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ProbeError(f"Refusing to overwrite existing probe artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            import numpy as np
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _score(y, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    cm = confusion_matrix(y, predicted, labels=CLASSES)
    recalls = np.divide(np.diag(cm), cm.sum(axis=1), out=np.full(4, np.nan), where=cm.sum(axis=1) > 0)
    if not np.isfinite(recalls).all():
        raise ProbeError("Metric labels must contain all four classes")
    return {"balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)), "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
        "per_class_recall": recalls.tolist(), "confusion_matrix": cm.tolist()}


def fit_probes(sources, cfg, directory):
    """Fit exactly four fixed CPU probes and persist states/probabilities atomically."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    train, validation, y_train, y_validation, u, v = _views(sources)
    factor_state = sources.legacy.factor_state
    train_views, validation_views = _feature_views(train, validation, factor_state, (u, v))
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("features_ordered", "features_window_mean", "core_ordered", "features_ordered_shuffled"):
        x_train = train_views["features_ordered"] if name == "features_ordered_shuffled" else train_views[name]
        x_validation = validation_views["features_ordered"] if name == "features_ordered_shuffled" else validation_views[name]
        fit_labels = y_train.copy()
        shuffle_seed = None
        if name == "features_ordered_shuffled":
            shuffle_seed = int(sources.seed) + 700001
            fit_labels = np.random.default_rng(shuffle_seed).permutation(y_train)
        scaler = StandardScaler().fit(x_train)
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
            raise ProbeError(f"{name} training scaler is invalid")
        model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
            tol=1e-4, random_state=0, class_weight=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(scaler.transform(x_train), fit_labels)
        converged = not any("ConvergenceWarning" in warning.category.__name__ for warning in caught)
        if not converged:
            raise ProbeError(f"{name} fixed logistic fit did not converge (max_iter=1000)")
        if not np.array_equal(model.classes_, np.asarray(CLASSES)):
            raise ProbeError(f"{name} fit returned invalid class identities")
        train_probability = model.predict_proba(scaler.transform(x_train))
        validation_probability = model.predict_proba(scaler.transform(x_validation))
        if not np.isfinite(train_probability).all() or not np.isfinite(validation_probability).all():
            raise ProbeError(f"{name} produced nonfinite probabilities")
        metadata = {"schema_name": "agfl-encoder-probe-v1", "name": name, "feature": VIEWS[name],
            "subject": int(sources.subject), "seed": int(sources.seed),
            "input_study_ids": {"baseline": sources.baseline_metadata.study_id,
                "legacy": sources.legacy_metadata.study_id},
            "input_source_sha256": {"baseline": sources.baseline_metadata.source_sha256,
                "legacy": sources.legacy_metadata.source_sha256},
            "input_manifest_sha256": {"baseline": sources.baseline_metadata.manifest_sha256,
                "legacy": sources.legacy_metadata.manifest_sha256},
            "input_artifact_sha256": {"baseline": sources.baseline_metadata.artifact_sha256,
                "legacy": sources.legacy_metadata.artifact_sha256},
            "split_ids": {"baseline": sources.baseline_metadata.split_id,
                "legacy": sources.legacy_metadata.split_id},
            "sample_ids": {"train": list(sources.legacy.train.sample_ids),
                "validation": list(sources.legacy.validation.sample_ids)},
            "fit": {"scaler": "StandardScaler", "scaler_partition": "train", "penalty": "l2",
                "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4,
                "random_state": 0, "class_weight": None, "converged": True,
                "iterations": model.n_iter_.tolist(), "fit_labels": "shuffled_train" if shuffle_seed is not None else "true_train",
                "shuffle_seed": shuffle_seed}}
        filename = target / f"{name}.npz"
        _atomic_npz(filename, mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_,
            intercept=model.intercept_, classes=model.classes_, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            train_probabilities=train_probability, validation_probabilities=validation_probability,
            train_ids=np.asarray(sources.legacy.train.sample_ids, dtype="U"),
            validation_ids=np.asarray(sources.legacy.validation.sample_ids, dtype="U"),
            train_labels=y_train, validation_labels=y_validation)
        probe = load_probe(filename)
        reloaded_train = probe.predict_proba(train_views["features_ordered"] if name in
            ("features_ordered", "features_ordered_shuffled") else train_views[name])
        reloaded_validation = probe.predict_proba(validation_views["features_ordered"] if name in
            ("features_ordered", "features_ordered_shuffled") else validation_views[name])
        if not np.allclose(reloaded_train, train_probability, rtol=1e-4, atol=1e-5) or not np.allclose(
                reloaded_validation, validation_probability, rtol=1e-4, atol=1e-5):
            raise ProbeError(f"{name} saved/reloaded probability mismatch")
        results[name] = {"path": str(filename), "converged": True,
            "train": _score(y_train, train_probability), "validation": _score(y_validation, validation_probability),
            "probability_reload_max_abs_error": float(max(np.max(np.abs(reloaded_train-train_probability)),
                np.max(np.abs(reloaded_validation-validation_probability)))), "probe": probe}
    return {"status": "complete", "synthetic": False, "partitions": ["train", "validation"],
        "subject": int(sources.subject), "seed": int(sources.seed), "probes": results,
        "ordered_probe": results["features_ordered"]["probe"],
        "factors_unchanged": True}


def load_probe(path):
    """Load a saved numeric probe without pickle and verify its basic schema."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("schema_name") != "agfl-encoder-probe-v1" or metadata.get("name") not in VIEWS:
                raise ProbeError("Unsupported probe metadata")
            mean, scale, coef = data["mean"].copy(), data["scale"].copy(), data["coef"].copy()
            intercept, classes = data["intercept"].copy(), data["classes"].copy()
    except ProbeError:
        raise
    except Exception as error:
        raise ProbeError(f"Cannot load probe {path}: {error}") from error
    if (mean.ndim != 1 or scale.shape != mean.shape or coef.shape != (4, len(mean)) or
            intercept.shape != (4,) or not np.array_equal(classes, CLASSES) or
            not all(np.isfinite(value).all() for value in (mean, scale, coef, intercept)) or (scale <= 0).any()):
        raise ProbeError("Persisted probe numeric state is malformed")
    return Probe(metadata["name"], mean, scale, coef, intercept, classes, metadata)
