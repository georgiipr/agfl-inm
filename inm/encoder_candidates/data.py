"""Paired train/validation data loading for the encoder-candidate study.

Historical split metadata is read and verified but never rewritten.  The public
object deliberately has no test partition or all-partition signal array.
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
from inm.data import load_subject


@dataclass(frozen=True)
class PartitionData:
    """One public partition, in canonical [N,22,4,250] layout."""

    raw: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]

    @property
    def signals(self) -> np.ndarray:
        """Compatibility spelling for the normalized raw signals."""
        return self.raw


@dataclass(frozen=True)
class PreparedData:
    """Training-fitted, paired data for one subject/seed; test data is absent."""

    subject: int
    seed: int
    train: PartitionData
    validation: PartitionData
    normalization: dict
    split_id: str
    data_id: str
    provenance: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read persisted input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"persisted input must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _loader_config(cfg: dict, subject: int) -> dict:
    prep = cfg["preprocessing"]
    return {
        "data_dir": cfg["data_dir"], "subjects": [int(subject)], "sessions": ["T"],
        "artifact_policy": "exclude", "filter_scope": "window",
        "window": int(prep["trial_samples"]), "offset_seconds": 0.0,
        "filter_window_samples": int(prep["window_samples"]),
        "lowcut": float(prep["filter_low_hz"]), "highcut": float(prep["filter_high_hz"]),
        "normalization": "none", "labels_dir": None,
    }


def _validate_bundle(bundle: SignalDataset, cfg: dict, subject: int) -> None:
    prep = cfg["preprocessing"]
    shape = (int(prep["channels"]), int(prep["trial_samples"]))
    if bundle.x.shape[1:] != shape:
        raise ValueError(f"Loaded signals must have [N,{shape[0]},{shape[1]}] shape")
    if not np.isfinite(bundle.x).all():
        raise ValueError("Loaded observations contain non-finite values")
    if bundle.metadata.get("subject") not in (None, f"A{subject:02d}"):
        raise ValueError("Loaded recording subject identity does not match requested subject")
    if bundle.metadata.get("modality") != "eeg" or bundle.metadata.get("num_classes") != 4:
        raise ValueError("Candidate data requires EEG metadata and exactly four classes")
    if float(bundle.metadata.get("sampling_rate", np.nan)) != float(prep["sampling_rate"]):
        raise ValueError("Loaded sampling rate does not match declared preprocessing")
    channels = bundle.metadata.get("channel_names")
    if channels != CHANNEL_NAMES or len(channels) != int(prep["channels"]):
        raise ValueError("Loaded channel order does not match canonical 22-electrode order")
    if len(set(bundle.sample_ids)) != len(bundle.sample_ids):
        raise ValueError("Loaded stable sample IDs are not unique")
    if not np.isin(bundle.y, np.arange(4)).all():
        raise ValueError("Labels must use all-four-class indices 0..3")
    sessions = bundle.metadata.get("sample_sessions")
    if sessions is not None and (len(sessions) != len(bundle) or set(sessions) != {"T"}):
        raise ValueError("Encoder-candidate data requires only T-session samples")
    actual_prep = bundle.metadata.get("preprocessing", {})
    expected = {"trial_samples": prep["trial_samples"],
                "window_samples": prep["window_samples"], "filter_low_hz": prep["filter_low_hz"],
                "filter_high_hz": prep["filter_high_hz"], "filter_scope": prep["filter_scope"]}
    for key, value in expected.items():
        actual_key = {"sampling_rate": "sampling_rate", "filter_scope": "filter_scope"}.get(key, key)
        # Legacy loader metadata uses window/filter_window_samples/lowcut/highcut.
        if key in {"trial_samples", "window_samples", "filter_low_hz", "filter_high_hz"}:
            actual_key = {"trial_samples": "window", "window_samples": "filter_window_samples",
                          "filter_low_hz": "lowcut", "filter_high_hz": "highcut"}.get(key, key)
        if actual_prep.get(actual_key) != value:
            raise ValueError(f"Loaded preprocessing mismatch for {key}: {actual_prep.get(actual_key)!r} != {value!r}")
    if actual_prep.get("offset_seconds") != 0.0 or actual_prep.get("artifact_policy") != "exclude":
        raise ValueError("Loaded cue offset or artifact exclusion policy differs from candidate protocol")
    skipped = bundle.metadata.get("skipped")
    if not isinstance(skipped, dict):
        raise ValueError("Loaded source exclusion counts are missing")


def _verify_metadata(cfg: dict, subject: int, seed: int, bundle: SignalDataset,
                     task_dir: Path) -> tuple[dict, dict, dict]:
    source_root = Path(cfg["split_source_dir"]) / "artifacts" / f"A{subject:02d}_seed_{seed}"
    split, split_hash = _read_json(source_root / "split.json")
    dataset, dataset_hash = _read_json(source_root / "dataset.json")
    manifest_path = Path(cfg["split_source_dir"]) / "study.json"
    manifest, manifest_hash = _read_json(manifest_path)

    if manifest.get("format") != "agfl-baseline-study-v1" or manifest.get("synthetic") is not False:
        raise ValueError("Split source manifest is not a real baselines-v1 study")
    if manifest.get("identity", {}).get("config", {}).get("protocol") != "within_session":
        raise ValueError("Persisted split source is not the declared within-session baseline study")
    if split.get("seed") != int(seed) or split.get("protocol") != "stratified":
        raise ValueError("Persisted split subject/seed or protocol does not match request")
    if split.get("subject_independent") is not False:
        raise ValueError("Persisted split must be the participant-specific stratified split")
    if split.get("fingerprint") != bundle.fingerprint or dataset.get("dataset_fingerprint") != bundle.fingerprint:
        raise ValueError("Loaded dataset identity differs from persisted baseline dataset metadata")
    if dataset.get("data_fingerprint") is None or dataset.get("provenance", {}).get("dataset_fingerprint") != bundle.fingerprint:
        raise ValueError("Persisted dataset metadata has inconsistent dataset identity")
    provenance = dataset.get("provenance", {})
    if provenance.get("subject") != int(subject) or provenance.get("seed") != int(seed):
        raise ValueError("Persisted dataset subject/seed provenance does not match request")
    if provenance.get("split_id") is None:
        raise ValueError("Persisted dataset provenance is missing split identity")
    if dataset.get("channel_names") != CHANNEL_NAMES:
        raise ValueError("Persisted channel order does not match canonical order")
    old_prep = provenance.get("config", {}).get("preprocessing", {})
    expected_old = {"channels": 22, "trial_samples": 1000, "windows": 4,
                    "window_samples": 250, "sampling_rate": 250,
                    "filter_low_hz": 2.0, "filter_high_hz": 30.0,
                    "filter_scope": "window"}
    if any(old_prep.get(key) != value for key, value in expected_old.items()):
        raise ValueError("Persisted dataset preprocessing does not match encoder-candidate contract")
    if provenance.get("source_exclusions") != dataset.get("source_exclusions"):
        raise ValueError("Persisted source exclusion counts conflict")
    if bundle.metadata.get("skipped") != dataset.get("source_exclusions"):
        raise ValueError("Loaded trial exclusions differ from persisted dataset metadata")

    identity = split.get("identity", {})
    expected_identity = {"version": "agfl-splits-v2", "fingerprint": bundle.fingerprint,
                         "seed": int(seed), "config": {"protocol": "stratified",
                         "train": 0.6, "validation": 0.2, "test": 0.2}}
    if identity != expected_identity:
        raise ValueError("Persisted split identity differs from expected subject/seed/fractions")
    if hashlib.sha256(canonical_json(identity).encode()).hexdigest() != split.get("split_id"):
        raise ValueError("Persisted split identity checksum is invalid")
    if provenance.get("split_id") != split.get("split_id"):
        raise ValueError("Persisted dataset and split identities conflict")
    normalization_stats = dataset.get("normalization_stats")
    if (not isinstance(normalization_stats, dict) or
            normalization_stats.get("convention") != "per_channel_train_trials_and_time_population_std" or
            len(normalization_stats.get("mean", [])) != 22 or
            len(normalization_stats.get("std", [])) != 22 or
            not np.isfinite(normalization_stats.get("mean", [])).all() or
            not np.isfinite(normalization_stats.get("std", [])).all() or
            any(value <= 0 for value in normalization_stats.get("std", []))):
        raise ValueError("Persisted baseline training normalization metadata is invalid")
    fingerprint_contents = {"provenance": provenance,
                           "normalization": normalization_stats,
                           "sample_ids": dataset.get("sample_ids")}
    if hashlib.sha256(canonical_json(fingerprint_contents).encode()).hexdigest() != dataset.get("data_fingerprint"):
        raise ValueError("Persisted dataset metadata fingerprint is invalid")
    partition_ids = split.get("sample_ids")
    if not isinstance(partition_ids, dict):
        raise ValueError("Persisted split is missing ordered partition sample IDs")
    if dataset.get("sample_ids") != bundle.sample_ids:
        raise ValueError("Loaded stable sample IDs differ from persisted dataset order")
    if len(bundle.sample_ids) != len(set(bundle.sample_ids)):
        raise ValueError("Loaded stable IDs contain duplicates")
    partitions = {name: split.get(name) for name in ("train", "validation", "test")}
    if any(not isinstance(part, list) or not part for part in partitions.values()):
        raise ValueError("Persisted split must contain nonempty train, validation, and held-out membership")
    sets = {}
    for name, indices in partitions.items():
        if any(type(index) is not int or index < 0 or index >= len(bundle) for index in indices):
            raise ValueError(f"Persisted {name} membership has invalid indices")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Persisted {name} membership contains repeated examples")
        sets[name] = set(indices)
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Persisted train, validation, and held-out membership overlap")
    if set.union(*sets.values()) != set(range(len(bundle))):
        raise ValueError("Persisted split does not cover exactly the loaded dataset")
    for name, indices in partitions.items():
        if partition_ids.get(name) != [bundle.sample_ids[i] for i in indices]:
            raise ValueError(f"Persisted ordered {name} IDs do not match loaded trial order")
        counts = np.bincount(bundle.y[indices], minlength=4).tolist()
        if dataset.get("class_counts", {}).get(name) != counts or split.get("class_counts", {}).get(name) != counts:
            raise ValueError(f"Persisted {name} class labels/counts differ from loaded trials")
        if not np.all(np.bincount(bundle.y[indices], minlength=4) > 0):
            raise ValueError(f"{name} partition lacks one or more classes")
    if any(indices != sorted(indices) for indices in partitions.values()):
        raise ValueError("Persisted partition membership order is not canonical")
    part_digest = hashlib.sha256(canonical_json(partitions).encode()).hexdigest()
    if split.get("partition_digest") != part_digest:
        raise ValueError("Persisted ordered partition membership checksum is invalid")

    sources = provenance.get("sources", [])
    subject_name = f"A{subject:02d}T.gdf"
    source = next((item for item in sources if item.get("name") == subject_name), None)
    manifest_inputs = manifest.get("input_sha256", {})
    recorded_hash = next((value for path, value in manifest_inputs.items()
                          if Path(path).name == subject_name), None)
    if source is None or recorded_hash is None or source.get("sha256") != recorded_hash:
        raise ValueError("Recording checksum is absent or inconsistent between metadata and study manifest")
    loaded_source = next((item for item in bundle.metadata.get("sources", [])
                          if item.get("name") == subject_name), None)
    if loaded_source != source:
        raise ValueError("Loaded recording checksum differs from persisted dataset provenance")

    historical_loader = provenance.get("loader_config", {})
    expected_loader = {"subjects": [int(subject)], "sessions": ["T"],
        "artifact_policy": "exclude", "filter_scope": "window", "window": 1000,
        "offset_seconds": 0.0, "filter_window_samples": 250,
        "lowcut": 2.0, "highcut": 30.0, "normalization": "none",
        "labels_dir": None}
    if any(historical_loader.get(key) != value for key, value in expected_loader.items()):
        raise ValueError("Persisted loader settings differ from the candidate preprocessing contract")
    historical_data_dir = historical_loader.get("data_dir")
    if not historical_data_dir or Path(historical_data_dir).resolve() != Path(cfg["data_dir"]).resolve():
        raise ValueError("Persisted loader data directory differs from configured recording input")

    hashes = {"split.json": split_hash, "dataset.json": dataset_hash,
              "study.json": manifest_hash}
    details = {"split": split, "dataset": dataset, "manifest": manifest,
               "metadata_sha256": hashes, "recording_sha256": {subject_name: source["sha256"]}}
    return details, hashes, partitions


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_subject(cfg: dict, subject: int, seed: int, directory: str | Path,
                    loader: Callable[[dict], SignalDataset] | None = None) -> PreparedData:
    """Load and verify persisted membership, then fit normalization on train only.

    ``loader`` receives the translated :func:`inm.data.load_subject` settings,
    providing a small synthetic fixture seam without changing split membership.
    """
    if subject not in cfg["subjects"]:
        raise ValueError(f"subject {subject} is not declared in config")
    if seed not in cfg["seeds"]:
        raise ValueError(f"seed {seed} is not declared in config")
    bundle = (loader or load_subject)(_loader_config(cfg, subject))
    _validate_bundle(bundle, cfg, subject)
    task_dir = Path(directory).expanduser().resolve()
    details, metadata_hashes, partitions = _verify_metadata(cfg, subject, seed, bundle, task_dir)
    split = details["split"]

    train_idx = np.asarray(partitions["train"], dtype=np.int64)
    val_idx = np.asarray(partitions["validation"], dtype=np.int64)
    raw = np.asarray(bundle.x, dtype=np.float32)
    train_physical = raw[train_idx].astype(np.float64)
    mean = train_physical.mean(axis=(0, 2), dtype=np.float64)
    std = train_physical.std(axis=(0, 2), ddof=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Training-only normalization statistics are non-finite")
    safe_std = np.maximum(std, 1e-8)

    def partition(indices: np.ndarray) -> PartitionData:
        physical = raw[indices].astype(np.float64)
        normalized = ((physical - mean[None, :, None]) / safe_std[None, :, None]).astype(np.float32)
        canonical = normalized.reshape(len(indices), 22, 4, 250).copy()
        labels = np.asarray(bundle.y[indices], dtype=np.int64).copy()
        ids = tuple(bundle.sample_ids[int(i)] for i in indices)
        return PartitionData(_readonly(canonical), _readonly(labels), ids)

    train, validation = partition(train_idx), partition(val_idx)
    normalization = {"convention": "per_channel_train_trials_and_time_population_std",
                     "mean": mean.tolist(), "std": safe_std.tolist(), "epsilon_floor": 1e-8,
                     "axes": ["training_trials", "time"]}
    provenance = {
        "subject": int(subject), "seed": int(seed), "split_id": split["split_id"],
        "dataset_fingerprint": bundle.fingerprint,
        "recording_sha256": details["recording_sha256"],
        "original_metadata_sha256": metadata_hashes,
        "partition_sample_ids": {"train": list(train.sample_ids),
                                  "validation": list(validation.sample_ids)},
        "partition_indices": {"train": partitions["train"],
                              "validation": partitions["validation"]},
        "normalization": normalization,
        "source_exclusions": details["dataset"].get("source_exclusions"),
        "channel_names": list(CHANNEL_NAMES),
        "preprocessing": cfg["preprocessing"],
        "filtered_dataset_fingerprint": bundle.fingerprint,
    }
    data_id = hashlib.sha256(canonical_json({
        "provenance": provenance,
        "train": {"ids": list(train.sample_ids), "labels": train.labels.tolist()},
        "validation": {"ids": list(validation.sample_ids), "labels": validation.labels.tolist()},
    }).encode()).hexdigest()
    provenance["data_id"] = data_id
    _atomic_json(task_dir / "data_provenance.json", provenance)
    return PreparedData(int(subject), int(seed), train, validation, normalization,
                        split["split_id"], data_id, provenance)
