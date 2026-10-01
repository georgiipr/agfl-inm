"""Dataset contract: finite signals [sample, channel, time] and stable identities."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_fingerprint(path: str | Path) -> dict:
    """Hash actual source bytes; paths and modification times are not identities."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def normalize_samples(x: np.ndarray, normalization: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3 or min(x.shape) < 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError('Normalization expects real, nonempty [N,C,T] signals')
    if not np.isfinite(x).all():
        raise ValueError('Normalization input contains non-finite signals')
    if normalization == "per_sample":
        # The original protocol standardizes each channel within its own trial.
        # No validation/test statistics enter a training sample.
        mean = x.mean(axis=-1, keepdims=True, dtype=np.float64)
        std = x.std(axis=-1, keepdims=True, dtype=np.float64)
        return ((x - mean) / (std + 1e-8)).astype(np.float32)
    if normalization in {"none", "train_channel"}:
        # train_channel is fitted by the study using training indices only.
        return x.astype(np.float32, copy=False)
    raise ValueError("normalization must be per_sample, train_channel, or none")


@dataclass
class SignalDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: list[str]
    metadata: dict
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float32)
        raw_labels = np.asarray(self.y)
        if not np.issubdtype(raw_labels.dtype, np.integer):
            if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
                raise ValueError("Labels must be finite integer class indices")
        self.y = np.ascontiguousarray(raw_labels, dtype=np.int64)
        self.groups = np.asarray(self.groups, dtype=str)
        self.sample_ids = [str(item) for item in self.sample_ids]
        if self.x.ndim != 3 or min(self.x.shape) < 1:
            raise ValueError("Dataset signals must have nonempty shape [samples, channels, time]")
        n, channels, samples = self.x.shape
        if self.y.shape != (n,) or self.groups.shape != (n,) or len(self.sample_ids) != n:
            raise ValueError("Signals, labels, subject groups, and sample IDs must have matching lengths")
        if len(set(self.sample_ids)) != n:
            raise ValueError("Sample IDs must be unique")
        if not np.isfinite(self.x).all():
            raise ValueError("Dataset contains non-finite signals after preprocessing")
        if not all(self.sample_ids) or not np.all(self.groups != ""):
            raise ValueError("Sample IDs and subject groups must be nonempty")
        num_classes = int(self.metadata.get("num_classes", self.y.max() + 1))
        if num_classes < 2 or self.y.min() < 0 or self.y.max() >= num_classes:
            raise ValueError("Labels must lie in [0, num_classes); classification needs at least two classes")
        if self.metadata.get("modality") != "eeg":
            raise ValueError("Dataset metadata must declare modality eeg")
        self.metadata.update(channels=channels, samples=samples, num_classes=num_classes,
                             num_samples=n, class_counts=np.bincount(self.y, minlength=num_classes).tolist())
        self._fingerprint = self._content_fingerprint()

    def _content_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (self.x, self.y):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(memoryview(array).cast("B"))
        digest.update(canonical_json(self.groups.tolist()).encode())
        digest.update(canonical_json(self.sample_ids).encode())
        # Absolute data paths do not alter identity when the same dataset is relocated.
        identity_metadata = dict(self.metadata)
        settings = dict(identity_metadata.get("preprocessing", {}))
        for key in ("data_dir", "path", "labels_dir"):
            settings.pop(key, None)
        identity_metadata["preprocessing"] = settings
        identity_metadata.pop("data_dir", None)
        digest.update(canonical_json(identity_metadata).encode())
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        # Recheck bytes if callers have modified arrays or metadata after loading.
        # The study works on a separate normalized data view.
        return self._content_fingerprint()

    def __len__(self):
        return len(self.y)


def bandpass_finite_spans(data: np.ndarray, fs: float, lowcut: float, highcut: float,
                          boundaries: list[int] | None = None, *, gdf_missing=False) -> tuple[np.ndarray, np.ndarray]:
    """Filter finite spans independently, never across run or missing-data gaps.

    Unusable samples remain NaN so trials touching them can be explicitly rejected.
    SOS filtering avoids the numerical sensitivity of high-order direct-form filters.
    This is an offline, zero-phase preprocessing protocol, not causal inference.
    """
    from scipy.signal import butter, sosfiltfilt

    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError(f"Require 0 < lowcut < highcut < Nyquist ({fs / 2:g} Hz)")
    signal = np.asarray(data, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("Filtering expects [channels, time]")
    valid = np.isfinite(signal).all(axis=0)
    # GDF may decode its 100 missing values as the negative digital limit instead
    # of NaN. A sustained simultaneous minimum across all EEG channels identifies
    # that sentinel; an isolated physiological minimum is retained.
    finite_signal = np.where(np.isfinite(signal), signal, np.inf)
    minima = finite_signal.min(axis=-1, keepdims=True)
    # This marker belongs to the GDF format. Other formats must not enable this heuristic.
    sentinel = valid & np.all(signal == minima, axis=0) if gdf_missing else np.zeros_like(valid)
    changes = np.diff(np.r_[False, sentinel, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if stop - start >= 100:
            valid[start:stop] = False
    stops = sorted(set([0, signal.shape[-1], *(boundaries or [])]))
    if stops[0] < 0 or stops[-1] > signal.shape[-1]:
        raise ValueError("Filter boundaries lie outside the recording")
    sos = butter(4, [lowcut, highcut], fs=fs, btype="bandpass", output="sos")
    filtered = np.full(signal.shape, np.nan, dtype=np.float64)
    for left, right in zip(stops[:-1], stops[1:]):
        edges = np.diff(np.r_[False, valid[left:right], False].astype(np.int8))
        starts = np.flatnonzero(edges == 1) + left
        ends = np.flatnonzero(edges == -1) + left
        for start, stop in zip(starts, ends):
            try:
                filtered[:, start:stop] = sosfiltfilt(sos, signal[:, start:stop], axis=-1)
            except ValueError as error:
                if "padlen" not in str(error):
                    raise
                # Very short segments cannot support this filter and remain NaN.
    return filtered, np.isfinite(filtered).all(axis=0)
