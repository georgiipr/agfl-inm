"""Data preparation for the separate baseline accuracy study.

Signals stay in physical filtered units in ``filtered_signals``.  ``signals``
uses a per-channel population mean and standard deviation fitted over training
trials and time only; the same values are then applied to every partition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from agfl.datasets.base import SignalDataset, canonical_json
from agfl.datasets.eeg import CHANNEL_NAMES
from agfl.datasets.splits import get_split
from inm.data import load_subject


@dataclass(frozen=True)
class PreparedData:
    """One subject/seed dataset with immutable partition membership."""

    signals: np.ndarray                 # [N, 22, 1000], train-channel normalized
    filtered_signals: np.ndarray        # [N, 22, 1000], unnormalized filtered units
    labels: np.ndarray                  # [N], class indices 0..3
    sample_ids: tuple[str, ...]
    split_indices: dict[str, np.ndarray]
    channel_names: tuple[str, ...]
    metadata: dict
    normalization_stats: dict[str, list[float]]
    data_fingerprint: str


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def _loader_config(cfg: dict, subject: int) -> dict:
    prep = cfg["preprocessing"]
    protocol = cfg["protocol"]
    labels_dir = cfg.get("external_labels_dir")
    if protocol == "cross_session" and not labels_dir:
        raise ValueError("cross_session requires external_labels_dir with official E-session labels")
    return {
        "data_dir": cfg["data_dir"],
        "subjects": [int(subject)],
        "sessions": ["T"] if protocol == "within_session" else ["T", "E"],
        "artifact_policy": "exclude",
        "filter_scope": "window",
        "window": int(prep["trial_samples"]),
        "offset_seconds": 0.0,
        "filter_window_samples": int(prep["window_samples"]),
        "lowcut": float(prep["filter_low_hz"]),
        "highcut": float(prep["filter_high_hz"]),
        # Fitting normalization belongs here, after the split is known.
        "normalization": "none",
        "labels_dir": labels_dir,
    }


def _validate_bundle(bundle: SignalDataset, cfg: dict, subject: int) -> None:
    prep = cfg["preprocessing"]
    expected_shape = (int(prep["channels"]), int(prep["trial_samples"]))
    if bundle.x.shape[1:] != expected_shape:
        raise ValueError(f"Loaded signals must have [N,{expected_shape[0]},{expected_shape[1]}] shape")
    if not np.isfinite(bundle.x).all():
        raise ValueError("Loaded observations contain non-finite values")
    if bundle.metadata.get("modality") != "eeg" or bundle.metadata.get("num_classes") != 4:
        raise ValueError("Baseline data requires EEG metadata and exactly four classes")
    fs = float(bundle.metadata.get("sampling_rate", np.nan))
    if not np.isfinite(fs) or fs != float(prep["sampling_rate"]):
        raise ValueError("Loaded sampling rate does not match preprocessing.sampling_rate")
    channel_names = bundle.metadata.get("channel_names")
    if channel_names != CHANNEL_NAMES or len(channel_names) != int(prep["channels"]):
        raise ValueError("Loaded channel order does not match the canonical 22-electrode order")
    if len(set(bundle.sample_ids)) != len(bundle.sample_ids):
        raise ValueError("Loaded sample IDs overlap")
    if not np.isin(bundle.y, np.arange(4)).all():
        raise ValueError("Labels must use all-four-class indices 0..3")
    sessions = bundle.metadata.get("sample_sessions")
    wanted = {"T"} if cfg["protocol"] == "within_session" else {"T", "E"}
    if sessions is not None:
        if len(sessions) != len(bundle) or not set(sessions) <= {"T", "E"}:
            raise ValueError("Session metadata is malformed")
        if set(sessions) != wanted:
            raise ValueError(f"{cfg['protocol']} requires exactly session(s) {sorted(wanted)}")
    elif cfg["protocol"] == "cross_session":
        raise ValueError("cross_session loader must preserve T/E session metadata")


def _config_identity(cfg: dict) -> dict:
    # Config path itself is provenance, while resolved data/output paths are
    # recorded for audit and do not change the bytes used to prepare a dataset.
    path = cfg.get("config_path")
    digest = None
    if path and Path(path).is_file():
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"config_path": path, "config_sha256": digest,
            "protocol": cfg["protocol"], "preprocessing": cfg["preprocessing"],
            "data_dir": cfg["data_dir"]}


def prepare_subject(cfg: dict, subject: int, seed: int, directory: str | Path,
                    *, loader: Callable[[dict], SignalDataset] | None = None) -> PreparedData:
    """Load, split, and normalize one subject without validation/test leakage.

    ``loader`` is an injectable seam for synthetic tests and smoke fixtures; it
    receives the translated legacy ``load_subject`` dictionary.
    """
    if subject not in cfg["subjects"]:
        raise ValueError(f"subject {subject} is not declared in config")
    loader_config = _loader_config(cfg, subject)
    bundle = (loader or load_subject)(loader_config)
    _validate_bundle(bundle, cfg, subject)

    split_settings = cfg["split"]
    if cfg["protocol"] == "within_session":
        split_cfg = {"protocol": "stratified", "train_fraction": split_settings["train"],
                     "validation_fraction": split_settings["validation"],
                     "test_fraction": split_settings["test"]}
    else:
        split_cfg = {"protocol": "session", "validation_within_train": split_settings["validation"]}
    output_dir = Path(directory).expanduser().resolve()
    split = get_split(bundle, split_cfg, int(seed), output_dir / ".splits")
    indices = {name: np.asarray(split[name], dtype=np.int64) for name in ("train", "validation", "test")}
    session_values = bundle.metadata.get("sample_sessions")
    if session_values is not None and cfg["protocol"] == "cross_session":
        sessions = np.asarray(session_values)
        if np.any(sessions[indices["train"]] != "T") or np.any(sessions[indices["validation"]] != "T") or \
                np.any(sessions[indices["test"]] != "E"):
            raise ValueError("cross_session partitions must be train/validation=T and test=E")
    for name, part in indices.items():
        counts = np.bincount(bundle.y[part], minlength=4)
        if not np.all(counts > 0):
            raise ValueError(f"{name} partition lacks classes required for four-class balanced accuracy: {counts.tolist()}")

    raw = np.asarray(bundle.x, dtype=np.float32).copy()
    train = raw[indices["train"]].astype(np.float64)
    mean = train.mean(axis=(0, 2))
    std = train.std(axis=(0, 2), ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Training-only normalization statistics are non-finite")
    safe_std = np.maximum(std, 1e-8)
    normalized = ((raw.astype(np.float64) - mean[None, :, None]) /
                  safe_std[None, :, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized observations contain non-finite values")

    stats = {"convention": "per_channel_train_trials_and_time_population_std",
             "mean": mean.tolist(), "std": safe_std.tolist(), "epsilon_floor": 1e-8}
    provenance = {"config": _config_identity(cfg), "loader_config": loader_config,
                  "sources": bundle.metadata.get("sources", []),
                  "source_exclusions": bundle.metadata.get("skipped", {}),
                  "subject": int(subject), "seed": int(seed), "dataset_fingerprint": bundle.fingerprint,
                  "split_id": split["split_id"]}
    fingerprint = hashlib.sha256(canonical_json({"provenance": provenance,
        "normalization": stats, "sample_ids": bundle.sample_ids}).encode()).hexdigest()
    metadata = {**bundle.metadata, "subject": int(subject), "seed": int(seed),
                "protocol": cfg["protocol"], "provenance": provenance,
                "class_counts": {name: np.bincount(bundle.y[idx], minlength=4).tolist()
                                 for name, idx in indices.items()},
                "source_exclusions": bundle.metadata.get("skipped", {}),
                "split_id": split["split_id"]}
    _atomic_json(output_dir / "dataset.json", {"data_fingerprint": fingerprint,
        "dataset_fingerprint": bundle.fingerprint, "provenance": provenance,
        "sample_ids": bundle.sample_ids, "class_counts": metadata["class_counts"],
        "source_exclusions": metadata["source_exclusions"], "channel_names": CHANNEL_NAMES,
        "normalization_stats": stats})
    _atomic_json(output_dir / "split.json", split)
    return PreparedData(signals=normalized, filtered_signals=raw, labels=bundle.y.copy(),
        sample_ids=tuple(bundle.sample_ids), split_indices=indices,
        channel_names=tuple(CHANNEL_NAMES), metadata=metadata,
        normalization_stats=stats, data_fingerprint=fingerprint)


def make_synthetic_signal_dataset(seed: int = 0, *, samples_per_class: int = 15,
                                  sessions: tuple[str, ...] = ("T",)) -> SignalDataset:
    """Build a small finite 22x1000 SignalDataset for smoke/tests; never real data."""
    if samples_per_class < 3:
        raise ValueError("samples_per_class must be at least three")
    if any(session not in {"T", "E"} for session in sessions) or len(set(sessions)) != len(sessions):
        raise ValueError("sessions must be a unique subset of T/E")
    rng = np.random.default_rng(seed)
    rows, labels, ids, sample_sessions = [], [], [], []
    for session in sessions:
        for label in range(4):
            for repeat in range(samples_per_class):
                rows.append(rng.normal(loc=label * 0.1, scale=1.0, size=(22, 1000)).astype(np.float32))
                labels.append(label)
                ids.append(f"synthetic:{session}:{label}:{repeat}")
                sample_sessions.append(session)
    return SignalDataset(np.stack(rows), np.asarray(labels),
        np.asarray(["A01"] * len(labels)), ids,
        {"dataset": "synthetic-eeg", "modality": "eeg", "num_classes": 4,
         "sampling_rate": 250.0, "channel_names": CHANNEL_NAMES[:],
         "label_names": ["left_hand", "right_hand", "feet", "tongue"],
         "preprocessing": {"synthetic": True}, "sources": [], "skipped": {},
         "sample_sessions": sample_sessions, "sample_runs": ["synthetic"] * len(labels),
         "synthetic": True})
