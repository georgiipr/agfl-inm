"""Low-cost full-input LOG-EUCLIDEAN covariance reference classifier.

Each trial's Ledoit--Wolf covariance is mapped through its symmetric matrix
logarithm and an isometric upper-triangle vectorization. This is not an
affine-invariant Riemannian-mean or tangent-space implementation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np


FORMAT = "agfl-baseline-covariance-v1"


def covariance_log_matrix(signal: np.ndarray, *, eigenvalue_floor: float = 1e-12) -> np.ndarray:
    """Return the symmetric log of a Ledoit--Wolf covariance for [C,T] data."""
    from sklearn.covariance import LedoitWolf

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Each trial must be a finite [channels,time] array with both dimensions >= 2")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be finite and positive")
    covariance = LedoitWolf(assume_centered=False, store_precision=False).fit(values.T).covariance_
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    trace = float(np.trace(covariance))
    if not np.isfinite(trace) or trace < 0:
        raise ValueError("Ledoit-Wolf covariance must have finite non-negative trace")
    floor = max(trace * eigenvalue_floor / covariance.shape[0], np.finfo(np.float64).tiny)
    log_values = np.log(np.maximum(eigenvalues, floor))
    logged = (eigenvectors * log_values[None, :]) @ eigenvectors.T
    return (logged + logged.T) * 0.5


def covariance_features(x: np.ndarray, *, eigenvalue_floor: float = 1e-12) -> np.ndarray:
    """Map [N,C,T] trials to weighted upper-triangle log-covariance features."""
    trials = np.asarray(x)
    if trials.ndim != 3 or trials.shape[0] < 1:
        raise ValueError("Signals must have shape [N,channels,time] with N >= 1")
    channels = trials.shape[1]
    upper = np.triu_indices(channels)
    weights = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    result = np.empty((len(trials), len(upper[0])), dtype=np.float64)
    for index, trial in enumerate(trials):
        result[index] = covariance_log_matrix(trial, eigenvalue_floor=eigenvalue_floor)[upper] * weights
    if not np.isfinite(result).all():
        raise FloatingPointError("Covariance feature extraction produced non-finite values")
    return result


class CovarianceClassifier:
    """Train-only StandardScaler and fixed-C multinomial-capable logistic model."""

    def __init__(self, *, eigenvalue_floor: float = 1e-12):
        if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be finite and positive")
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.scaler_ = None
        self.classifier_ = None
        self.classes_ = None

    def fit(self, x, y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        features = covariance_features(x, eigenvalue_floor=self.eigenvalue_floor)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(features) or len(np.unique(labels)) != 4:
            raise ValueError("Expected one label per trial and all four training classes")
        if "l1_ratio" not in LogisticRegression().get_params():
            raise RuntimeError("Installed scikit-learn LogisticRegression lacks explicit l1_ratio support")
        # lbfgs supports multinomial loss; omitting multi_class is deliberate
        # because sklearn 1.9 removed that deprecated parameter. l1_ratio=0
        # states L2 explicitly across the supported scikit-learn API range.
        self.scaler_ = StandardScaler(copy=True, with_mean=True, with_std=True).fit(features)
        self.classifier_ = LogisticRegression(
            C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=1000,
            tol=1e-4, random_state=0,
        ).fit(self.scaler_.transform(features), labels)
        if len(self.classifier_.classes_) != 4:
            raise ValueError("LogisticRegression did not fit all four classes")
        self.classes_ = self.classifier_.classes_.copy()
        return self

    def predict_proba(self, x) -> np.ndarray:
        if self.scaler_ is None or self.classifier_ is None:
            raise RuntimeError("CovarianceClassifier must be fitted or loaded before prediction")
        features = covariance_features(x, eigenvalue_floor=self.eigenvalue_floor)
        if features.shape[1] != len(self.scaler_.mean_):
            raise ValueError("Prediction channel count differs from the fitted model")
        probabilities = self.classifier_.predict_proba(self.scaler_.transform(features))
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Covariance probabilities are non-finite")
        return probabilities

    def save(self, path) -> Path:
        """Write a non-pickle numeric NPZ checkpoint suitable for trusted local use."""
        if self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("Cannot save an unfitted CovarianceClassifier")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format": FORMAT, "eigenvalue_floor": self.eigenvalue_floor,
                    "class_order": self.classes_.tolist(), "C": 1.0,
                    "penalty": "l2", "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000,
                    "tol": 1e-4, "random_state": 0}
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                                  dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)),
                    scaler_mean=self.scaler_.mean_, scaler_scale=self.scaler_.scale_,
                    scaler_var=self.scaler_.var_, scaler_n_samples=np.asarray(self.scaler_.n_samples_seen_),
                    coefficients=self.classifier_.coef_, intercept=self.classifier_.intercept_,
                    classes=self.classifier_.classes_)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    @classmethod
    def load(cls, path) -> "CovarianceClassifier":
        """Restore the numeric NPZ state without executing serialized objects."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        with np.load(path, allow_pickle=False) as state:
            metadata = json.loads(str(state["metadata"].item()))
            if metadata.get("format") != FORMAT:
                raise ValueError("Unsupported covariance checkpoint format")
            result = cls(eigenvalue_floor=float(metadata["eigenvalue_floor"]))
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            scaler.mean_ = state["scaler_mean"].copy()
            scaler.scale_ = state["scaler_scale"].copy()
            scaler.var_ = state["scaler_var"].copy()
            scaler.n_samples_seen_ = state["scaler_n_samples"].copy()
            scaler.n_features_in_ = len(scaler.mean_)
            model = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=1000,
                                       tol=1e-4, random_state=0)
            model.classes_ = state["classes"].copy()
            model.coef_ = state["coefficients"].copy()
            model.intercept_ = state["intercept"].copy()
            model.n_features_in_ = model.coef_.shape[1]
            model.n_iter_ = np.zeros(len(model.classes_), dtype=np.int32)
            if model.classes_.tolist() != metadata.get("class_order"):
                raise ValueError("Covariance checkpoint class mapping is inconsistent")
            result.scaler_, result.classifier_, result.classes_ = scaler, model, model.classes_.copy()
        return result
