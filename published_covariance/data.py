"""Audited EEG-ATCNet BCI IV-2a data boundary.

The native source is pinned in docs/published-covariance/checkpoint-audit.json.
This module intentionally implements only its BCI MAT recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


CHANNEL_IDS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
CLASS_NAMES = ("left", "right", "feet", "tongue")
NORMALIZATION_SHUFFLE_SEED = 42
NATIVE_SLICE = (375, 1500)
SCHEMA_VERSION = 1
_SUBJECT_RE = re.compile(r"A0[1-9]\Z")
_TRIAL_ID_RE = re.compile(r"(A0[1-9]):T:r\d{2}:t\d{3}\Z")


def _finite_integer(value: Any, name: str) -> int:
    """Validate a native scalar before applying the author's integer cast."""
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError(f"native {name} must be one finite integer scalar")
    try:
        number = float(scalar.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"native {name} must be one finite integer scalar") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"native {name} must be one finite integer scalar")
    return int(number)


def _validated_normalization_identity(subject: str, source_sha256: str,
                                      trial_ids: tuple[str, ...] | list[str]) -> tuple[str, list[str]]:
    normalized_subject = str(subject).upper()
    if not _SUBJECT_RE.fullmatch(normalized_subject):
        raise ValueError("subject must be A01 through A09")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("source_sha256 must be a lowercase 64-character SHA-256 digest")
    ids = [str(value) for value in trial_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("trial_ids must be nonempty and unique")
    for trial_id in ids:
        match = _TRIAL_ID_RE.fullmatch(trial_id)
        if match is None or match.group(1) != normalized_subject:
            raise ValueError("normalization trial IDs must identify this subject's T session")
    return normalized_subject, ids


def _validate_saved_ids(metadata: dict[str, Any]) -> None:
    subject = metadata.get("subject")
    source_hash = metadata.get("source_sha256")
    normalized_subject, ids = _validated_normalization_identity(
        subject, source_hash, metadata.get("fit_trial_ids", ()))
    if normalized_subject != subject:
        raise ValueError("normalization subject identity is not canonical")
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if metadata.get("fit_trial_ids_sha256") != digest:
        raise ValueError("normalization trial ID hash mismatch")


@dataclass(frozen=True)
class NativeRecording:
    """Native trials in chronological source order, shaped [N,22,1125]."""

    values: np.ndarray
    labels: np.ndarray
    trial_ids: tuple[str, ...]
    run_ids: np.ndarray
    artifacts: np.ndarray
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(record: np.void, name: str) -> Any:
    try:
        value = record[name]
        # MATLAB struct fields are already their native arrays here. In
        # particular trial/y/artifacts may be [0,1] for calibration-only runs.
        return value
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"native MAT run is missing a usable {name!r} field") from exc


def load_native_recording(path: str | Path, subject: str, session: str) -> NativeRecording:
    """Read one author-format AxxT/AxxE MAT file without dropping artifacts.

    Stored ``a_trial`` values are used directly as Python slice starts. This is
    the author's indexing convention (one sample after the GDF cue marker).
    """
    source = Path(path)
    subject = str(subject).upper()
    session = str(session).upper()
    if subject not in {f"A{i:02d}" for i in range(1, 10)}:
        raise ValueError("subject must be A01 through A09")
    if session not in {"T", "E"}:
        raise ValueError("session must be explicitly 'T' or 'E'")
    if source.stem.upper() != subject + session:
        raise ValueError("MAT filename must identify the requested subject and session")
    if not source.is_file():
        raise FileNotFoundError(source)

    mat = loadmat(source, struct_as_record=True, squeeze_me=False)
    if "data" not in mat:
        raise ValueError("expected author-format MAT variable 'data'; label-only MAT is unsupported")
    runs = mat["data"]
    if runs.ndim != 2 or runs.shape[0] != 1:
        raise ValueError("native 'data' must be the author's row of run structs")

    chunks: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[str] = []
    run_ids: list[int] = []
    artifacts: list[bool] = []
    run_metadata: list[dict[str, Any]] = []
    chronological_id = 0
    for run_index in range(runs.size):
        record = runs[0, run_index][0, 0]
        continuous = np.asarray(_field(record, "X"))
        starts = np.asarray(_field(record, "trial")).reshape(-1)
        raw_labels = np.asarray(_field(record, "y")).reshape(-1)
        flags = np.asarray(_field(record, "artifacts")).reshape(-1)
        sampling = np.asarray(_field(record, "fs")).reshape(-1)
        if continuous.ndim != 2 or continuous.shape[1] < 22:
            raise ValueError(f"run {run_index + 1} must contain at least 22 ordered channels")
        if not (len(starts) == len(raw_labels) == len(flags)):
            raise ValueError(f"run {run_index + 1} trial, label and artifact counts differ")
        if sampling.size != 1 or _finite_integer(sampling[0], "fs") != 250:
            raise ValueError(f"run {run_index + 1} must declare 250 Hz")
        run_metadata.append({"run_index_1based": run_index + 1,
                             "continuous_shape": list(continuous.shape),
                             "trial_count": int(len(starts)), "sampling_rate_hz": 250})
        for within_run, (stored_start, raw_label, artifact) in enumerate(zip(starts, raw_labels, flags), 1):
            # Validate first, then mirror the pinned author's integer values.
            start = _finite_integer(stored_start, "start")
            label = _finite_integer(raw_label, "label")
            flag = _finite_integer(artifact, "artifacts")
            trial_id = f"{subject}:{session}:r{run_index + 1:02d}:t{within_run:03d}"
            chronological_id += 1
            begin = start + NATIVE_SLICE[0]
            end = start + NATIVE_SLICE[1]
            if label not in (1, 2, 3, 4):
                raise ValueError(f"{trial_id} has noncanonical one-based class {label}")
            # The native loader first materializes X[a_trial:a_trial+1750],
            # then crops [375:1500]. Require that outer window to exist.
            if start < 0 or start + 1750 > continuous.shape[0]:
                raise ValueError(f"{trial_id} does not have the native 1750-sample outer window")
            # all_trials=True in the author's function: flagged trials remain.
            chunks.append(continuous[begin:end, :22].T.copy())
            labels.append(label - 1)
            trial_ids.append(trial_id)
            run_ids.append(run_index + 1)
            artifacts.append(flag != 0)

    if not chunks:
        raise ValueError("MAT contains no complete native trials")
    values = np.stack(chunks)
    label_array = np.asarray(labels, dtype=np.int64)
    run_array = np.asarray(run_ids, dtype=np.int64)
    artifact_array = np.asarray(artifacts, dtype=np.bool_)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "subject": subject,
        "session": session,
        "source_format": "EEG-ATCNet structured MAT data/run struct",
        "sampling_rate_hz": 250,
        "units": "microvolts (native source values; no conversion)",
        "recording_channels": 25,
        "channel_ids": list(CHANNEL_IDS),
        "channel_selection": "first 22 columns in source order",
        "class_names_by_zero_based_label": list(CLASS_NAMES),
        "label_source": "a_y, one-based 1..4 converted to zero-based 0..3",
        "trial_start": "stored a_trial integer used directly as Python start; no +/-1 adjustment",
        "slice_from_stored_start": [NATIVE_SLICE[0], NATIVE_SLICE[1]],
        "input_shape_per_trial": [22, NATIVE_SLICE[1] - NATIVE_SLICE[0]],
        "filtering": "none", "rereference": "none", "resampling": "none",
        "artifact_policy": "all_trials=True; retain nonzero artifact flags",
        "trial_order": "native run order then within-run annotation order",
        "trial_id_encoding": "{subject}:{session}:r{run_index_1based:02d}:t{within_run_index_1based:03d}",
        "chronological_id_count": chronological_id,
        "included_count": len(trial_ids),
        "excluded_trials": [],
        "included_artifact_count": int(artifact_array.sum()),
        "runs": run_metadata,
        "normalization": "not applied by loader; fit/replay explicitly via native normalization helpers",
        "e_session_access": "only when caller explicitly requests session='E'; no implicit E load",
    }
    return NativeRecording(values, label_array, tuple(trial_ids), run_array, artifact_array, metadata)


def fit_native_normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recreate author's shuffle(42) then per-channel/per-timepoint scaler fit."""
    x = np.asarray(values)
    if x.ndim != 3 or x.shape[1] != 22 or x.shape[2] != 1125 or x.shape[0] < 1:
        raise ValueError("values must have shape [N,22,1125] with N >= 1")
    if not np.isfinite(x).all():
        raise ValueError("normalization fit values must be finite")
    order = shuffle(np.arange(len(x)), random_state=NORMALIZATION_SHUFFLE_SEED)
    mean = np.empty((22, 1125), dtype=np.float64)
    scale = np.empty_like(mean)
    for channel in range(22):
        scaler = StandardScaler().fit(x[order, channel, :])
        mean[channel] = scaler.mean_
        scale[channel] = scaler.scale_
    return mean, scale


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
    """Apply saved scaler safely; hidden channel values are selected out first."""
    x = np.asarray(values)
    mu = np.asarray(mean)
    sigma = np.asarray(scale)
    observed = np.asarray(mask)
    if x.ndim != 3 or x.shape[1:] != (22, 1125):
        raise ValueError("values must have shape [N,22,1125]")
    if mu.shape != (22, 1125) or sigma.shape != (22, 1125):
        raise ValueError("mean and scale must both have shape [22,1125]")
    if observed.dtype != np.bool_ or observed.shape != (len(x), 22):
        raise ValueError("mask must be boolean with shape [N,22]")
    if not observed.any(axis=1).all():
        raise ValueError("all-missing trials are forbidden")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("normalization statistics must be finite with positive scales")
    safe = np.where(observed[..., None], x, np.zeros((), dtype=x.dtype))
    if not np.isfinite(safe[observed[..., None].repeat(1125, axis=2)]).all():
        raise ValueError("observed values must be finite")
    normalized = (safe - mu[None, :, :]) / sigma[None, :, :]
    return np.where(observed[..., None], normalized, np.zeros((), dtype=normalized.dtype))


def stratified_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Contract split: class-ordered PCG64 permutations, 20% floor to validation."""
    y = np.asarray(labels)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not len(y):
        raise ValueError("labels must be a nonempty integer vector")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isin(y, np.arange(4)).all():
        raise ValueError("labels must use frozen class IDs 0..3")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    validation: list[int] = []
    for class_id in range(4):
        indices = np.flatnonzero(y == class_id)
        n_val = max(1, int(np.floor(0.2 * len(indices)))) if len(indices) else 0
        if len(indices) - n_val < 1:
            raise ValueError(f"class {class_id} would have no training trial")
        validation.extend(rng.permutation(indices)[:n_val].tolist())
    val = np.asarray(sorted(validation), dtype=np.int64)
    is_val = np.zeros(len(y), dtype=bool)
    is_val[val] = True
    train = np.flatnonzero(~is_val).astype(np.int64)
    return train, val


def save_normalization(path: str | Path, mean: np.ndarray, scale: np.ndarray,
                       *, subject: str, source_sha256: str,
                       trial_ids: tuple[str, ...] | list[str]) -> None:
    """Persist exact scaler arrays with their T source and trial identity."""
    mu, sigma = np.asarray(mean), np.asarray(scale)
    if (mu.shape != (22, 1125) or sigma.shape != mu.shape or
            not np.issubdtype(mu.dtype, np.number) or not np.issubdtype(sigma.dtype, np.number) or
            not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any()):
        raise ValueError("invalid normalization arrays")
    normalized_subject, ids = _validated_normalization_identity(subject, source_sha256, trial_ids)
    metadata = {"schema_version": SCHEMA_VERSION, "subject": normalized_subject,
                "fit_session": "T", "fit_trial_ids": ids,
                "fit_trial_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "source_sha256": source_sha256, "shuffle_seed": NORMALIZATION_SHUFFLE_SEED,
                "transform": "per-channel/per-timepoint sklearn StandardScaler; ddof=0 scale",
                "mean_shape": [22, 1125], "scale_shape": [22, 1125]}
    destination = Path(path)
    # Create the exact destination exclusively so even concurrent callers
    # cannot replace a prior provenance record.
    with destination.open("xb") as handle:
        np.savez_compressed(handle, mean=mu, scale=sigma,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_normalization(path: str | Path, *, subject: str | None = None,
                       source_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a saved T scaler and optionally bind it to expected study inputs."""
    with np.load(Path(path), allow_pickle=False) as saved:
        if set(saved.files) != {"mean", "scale", "metadata"}:
            raise ValueError("unexpected normalization archive fields")
        mean, scale = saved["mean"].copy(), saved["scale"].copy()
        metadata = json.loads(str(saved["metadata"].item()))
    if (not isinstance(metadata, dict) or metadata.get("fit_session") != "T" or
            type(metadata.get("schema_version")) is not int or metadata.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("normalization archive is not a supported T-fitted scaler")
    _validate_saved_ids(metadata)
    if subject is not None and metadata.get("subject") != str(subject).upper():
        raise ValueError("normalization subject mismatch")
    if source_sha256 is not None and metadata.get("source_sha256") != source_sha256:
        raise ValueError("normalization source hash mismatch")
    if (mean.shape != (22, 1125) or scale.shape != mean.shape or
            not np.issubdtype(mean.dtype, np.number) or not np.issubdtype(scale.dtype, np.number) or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("invalid saved normalization arrays")
    return mean, scale, metadata
