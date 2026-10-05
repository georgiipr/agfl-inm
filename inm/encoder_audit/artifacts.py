"""Verified readers for the historical studies, exposing train/validation only.

This module never calls either study's writer or split initializer. Labels are
recovered from the canonical EEG loader and joined by stable sample ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inventory import REPLAY_PATHS
from .protocol import ROOT, digest, file_sha256


class ArtifactError(ValueError):
    """An input artifact is corrupt, mismatched, unsupported, or unsafe to use."""


@dataclass(frozen=True)
class SourceMetadata:
    study_id: str
    source_sha256: str
    manifest_sha256: str
    task_path: str
    subject: int
    seed: int
    split_id: str
    dataset_fingerprint: str
    artifact_sha256: dict[str, str]
    replay_source_status: dict[str, str]


@dataclass(frozen=True)
class PartitionView:
    sample_ids: tuple[str, ...]
    labels: Any


@dataclass(frozen=True)
class LegacyView:
    train: PartitionView
    validation: PartitionView
    train_features: Any
    validation_features: Any
    raw_mean: Any
    raw_std: Any
    feature_mean: Any
    feature_std: Any
    encoder_state: dict[str, Any]
    factor_state: dict[str, Any]
    encoder_selection: dict[str, Any]
    original_heads: dict[str, Any]


@dataclass(frozen=True)
class BaselineCheckpoint:
    arm: str
    selected_epoch: int
    selection_policy: str
    model_type: str
    constructor: dict[str, Any]
    state_dict: dict[str, Any]
    metadata: dict[str, Any]
    history_selection: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BaselineView:
    train: PartitionView
    validation: PartitionView
    normalization_mean: Any
    normalization_std: Any
    checkpoints: dict[str, BaselineCheckpoint]


@dataclass(frozen=True)
class TaskSources:
    subject: int
    seed: int
    baseline_metadata: SourceMetadata
    legacy_metadata: SourceMetadata
    baseline: BaselineView
    legacy: LegacyView


LEGACY_CHANNELS = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
                   "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz")
LABEL_NAMES = ("left_hand", "right_hand", "feet", "tongue")


def _json(path: Path):
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactError(f"Invalid or missing JSON artifact {path}: {error}") from error


def _verified(path: Path, expected: str) -> str:
    if not path.is_file():
        raise ArtifactError(f"Required artifact is missing: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactError(f"Missing valid recorded SHA-256 for {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ArtifactError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
    return actual


def _source_status(study: str, source_map: dict) -> dict[str, str]:
    status = {}
    for relative in REPLAY_PATHS[study]:
        expected = source_map.get(relative)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            status[relative] = "missing"
        elif not isinstance(expected, str) or file_sha256(path) != expected:
            status[relative] = "mismatch"
        else:
            status[relative] = "match"
    failed = [name for name, value in status.items() if value != "match"]
    if failed:
        raise ArtifactError(f"Historical {study} replay source mismatch: {', '.join(failed)}")
    return status


def _loader_labels(cfg: dict, subject: int) -> tuple[dict[str, int], tuple[str, ...]]:
    # Lazy by design: plan and inventory remain stdlib-only.
    from inm.data import load_subject
    prep = cfg["preprocessing"]
    bundle = load_subject({"data_dir": cfg["data_dir"], "subjects": [subject], "sessions": ["T"],
        "artifact_policy": "exclude", "filter_scope": "window", "window": prep["trial_samples"],
        "offset_seconds": 0.0, "filter_window_samples": prep["window_samples"],
        "lowcut": prep["filter_low_hz"], "highcut": prep["filter_high_hz"],
        "normalization": "none", "labels_dir": None})
    if tuple(bundle.metadata.get("channel_names", ())) != LEGACY_CHANNELS:
        raise ArtifactError("Loaded channel order differs from historical canonical order")
    if tuple(bundle.metadata.get("label_names", ())) != LABEL_NAMES:
        raise ArtifactError("Loaded four-class mapping differs from historical mapping")
    if bundle.x.ndim != 3 or bundle.x.shape[1:] != (22, 1000):
        raise ArtifactError("Canonical EEG loader returned an unsupported signal shape")
    labels = bundle.y
    if (len(labels) != len(bundle.sample_ids) or len(set(bundle.sample_ids)) != len(bundle.sample_ids) or
            set(map(int, labels)) != set(range(4))):
        raise ArtifactError("Canonical EEG loader returned duplicate IDs or invalid class labels")
    return dict(zip(bundle.sample_ids, map(int, labels))), tuple(bundle.sample_ids)


def _verify_recording(cfg: dict, subject: int, baseline_manifest: dict,
                      legacy_dataset: dict) -> str:
    path = Path(cfg["data_dir"]) / f"A{subject:02d}T.gdf"
    if not path.is_file():
        raise ArtifactError(f"Required verified recording is missing: {path}")
    actual = file_sha256(path)
    input_map = baseline_manifest.get("input_sha256", {})
    recorded = input_map.get(str(path)) if isinstance(input_map, dict) else None
    if not isinstance(recorded, str) or actual != recorded:
        raise ArtifactError(f"Recording checksum differs from baseline manifest: {path}")
    sources = legacy_dataset.get("sources", [])
    matches = [row for row in sources if isinstance(row, dict) and row.get("name") == path.name]
    if len(matches) != 1 or matches[0].get("sha256") != actual:
        raise ArtifactError(f"Recording checksum differs from legacy dataset metadata: {path}")
    return actual


def _partition(split: dict, dataset: dict, subject: int, seed: int,
               label_by_id: dict[str, int], sample_ids: tuple[str, ...]):
    identity = split.get("identity", {})
    if split.get("seed") != seed or identity.get("seed") != seed:
        raise ArtifactError("Saved split seed does not match requested task")
    if identity.get("protocol") != "stratified" or split.get("protocol") != "stratified":
        raise ArtifactError("Unsupported saved split protocol")
    if split.get("fingerprint") != identity.get("fingerprint"):
        raise ArtifactError("Split fingerprint and split identity disagree")
    stored = split.get("sample_ids")
    partitions = {}
    index_sets = {}
    total = len(sample_ids)
    for part in ("train", "validation", "test"):
        indices, ids = split.get(part), stored.get(part) if isinstance(stored, dict) else None
        if (not isinstance(indices, list) or not indices or
                any(type(i) is not int or i < 0 or i >= total for i in indices) or
                len(set(indices)) != len(indices)):
            raise ArtifactError(f"Invalid or out-of-bounds {part} split indices")
        if (not isinstance(ids, list) or len(ids) != len(indices) or len(set(ids)) != len(ids) or
                ids != [sample_ids[i] for i in indices]):
            raise ArtifactError(f"Saved {part} IDs do not match authoritative loader order")
        if any(sid not in label_by_id for sid in ids):
            raise ArtifactError(f"Cannot recover {part} labels by stable sample ID")
        index_sets[part] = set(indices)
        # Test data is checked for membership only. No test labels or sample view
        # are constructed or copied into the public object.
        if part != "test":
            partitions[part] = PartitionView(tuple(ids), _array([label_by_id[sid] for sid in ids], "int64"))
    if any(index_sets[a] & index_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ArtifactError("Saved partitions overlap")
    if len(set().union(*index_sets.values())) != total:
        raise ArtifactError("Saved partitions do not cover the verified dataset")
    channels = dataset.get("channel_names")
    if channels is not None and tuple(channels) != LEGACY_CHANNELS:
        raise ArtifactError("Saved dataset channel order differs from canonical order")
    return partitions, index_sets


def _array(value, dtype=None):
    import numpy as np
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ArtifactError("Nonfinite values in exposed source artifact")
    return array


def _torch_load(path: Path):
    import inspect
    import torch
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise ArtifactError(f"Installed torch lacks safe weights_only loading required for {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ArtifactError(f"Unsupported or unsafe tensor checkpoint {path}: {error}") from error


def _check_finite_tree(value: Any, label: str) -> None:
    import numpy as np
    if hasattr(value, "detach") and hasattr(value, "is_floating_point"):
        if value.is_floating_point() and not bool(value.isfinite().all()):
            raise ArtifactError(f"Nonfinite tensor in {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, np.ndarray) and not np.isfinite(value).all():
        raise ArtifactError(f"Nonfinite values in {label}")


def _source_metadata(study, cfg, subject, seed, manifest, task_dir, dataset, split,
                     artifact_sha256, status):
    identity = manifest["identity"] if study == "baseline" else manifest["source"]
    study_id = digest(identity) if study == "baseline" else manifest["study_id"]
    return SourceMetadata(study_id=study_id,
        source_sha256=(identity.get("source_sha256") if study == "baseline" else identity.get("source_sha256")),
        manifest_sha256=file_sha256(Path(cfg[f"{study}_dir"]) / "study.json"),
        task_path=str(task_dir), subject=subject, seed=seed, split_id=split["split_id"],
        dataset_fingerprint=dataset.get("data_fingerprint", split["fingerprint"]),
        artifact_sha256=artifact_sha256, replay_source_status=status)


def load_task_sources(cfg: dict, subject: int, seed: int) -> TaskSources:
    """Load one verified task. Public sample arrays contain train and validation only."""
    if (subject, seed) not in [(s, z) for s in cfg["subjects"] for z in cfg["seeds"]]:
        raise ArtifactError("Requested task is outside the configured subject/seed matrix")
    manifests = {study: _json(Path(cfg[f"{study}_dir"]) / "study.json") for study in ("baseline", "legacy")}
    baseline_manifest, legacy_manifest = manifests["baseline"], manifests["legacy"]
    base_identity = baseline_manifest.get("identity", {})
    if not isinstance(base_identity.get("source_sha256"), dict):
        raise ArtifactError("Baseline manifest lacks its historical source map")
    recorded_config = base_identity.get("config", {}).get("config_path")
    if not isinstance(recorded_config, str):
        raise ArtifactError("Baseline manifest lacks its historical config path")
    _verified(Path(recorded_config), base_identity.get("config_sha256"))
    legacy_source_record = legacy_manifest.get("source", {})
    if (legacy_manifest.get("study_id") != digest({"config": legacy_manifest.get("config"),
            "source": legacy_source_record}) or
            legacy_source_record.get("source_sha256") != digest(legacy_source_record.get("files_sha256", {}))):
        raise ArtifactError("Legacy manifest study/source identity is inconsistent")
    base_dir = Path(cfg["baseline_dir"]) / "artifacts" / f"A{subject:02d}_seed_{seed}"
    legacy_dir = Path(cfg["legacy_dir"]) / "artifacts" / f"A{subject:02d}_seed_{seed}"
    base_dataset, base_split = _json(base_dir / "dataset.json"), _json(base_dir / "split.json")
    legacy_dataset, legacy_split = _json(legacy_dir / "dataset.json"), _json(legacy_dir / "split.json")
    for study, manifest in manifests.items():
        source_map = (manifest["identity"].get("source_sha256", {}) if study == "baseline"
                      else manifest.get("source", {}).get("files_sha256", {}))
        _source_status(study, source_map)
    recording_hash = _verify_recording(cfg, subject, baseline_manifest, legacy_dataset)
    label_by_id, loaded_ids = _loader_labels(cfg, subject)
    if any(not sample_id.startswith(f"A{subject:02d}T:cue:") for sample_id in loaded_ids):
        raise ArtifactError("Canonical EEG loader returned a sample from another subject/session")
    base_parts, base_indices = _partition(base_split, base_dataset, subject, seed, label_by_id, loaded_ids)
    legacy_parts, legacy_indices = _partition(legacy_split, legacy_dataset, subject, seed, label_by_id, loaded_ids)
    # Cross-study pairing is an analysis decision, not an input-integrity
    # requirement. Keep each study's independently verified split intact; the
    # alignment audit marks a contrast unavailable when these IDs differ.
    baseline_id = digest(baseline_manifest["identity"])
    legacy_id = legacy_manifest.get("study_id")
    # Verify task-level data identities before checkpoint deserialization.
    if base_dataset.get("channel_names") and tuple(base_dataset["channel_names"]) != LEGACY_CHANNELS:
        raise ArtifactError("Baseline channel order is not canonical")
    for task_dir, split, dataset, expected_id in ((base_dir, base_split, base_dataset, baseline_id),
                                                   (legacy_dir, legacy_split, legacy_dataset, legacy_id)):
        if split.get("seed") != seed or dataset.get("subject", subject) != subject:
            raise ArtifactError("Swapped subject/seed task identity")
        if split.get("split_id") is None or not split.get("fingerprint"):
            raise ArtifactError("Missing saved split identity")
        if task_dir == legacy_dir:
            calibration_meta = _json(task_dir / "calibration.json")
            wanted = {"subject": subject, "seed": seed, "study_id": expected_id,
                      "split_id": split["split_id"], "dataset_fingerprint": split["fingerprint"]}
            if any(calibration_meta.get("identity", {}).get(k) != v for k, v in wanted.items()):
                raise ArtifactError("Legacy calibration identity does not match requested task")
    # Verify the legacy cache hash before any torch deserialization.
    cal_meta = _json(legacy_dir / "calibration.json")
    legacy_pt = legacy_dir / "calibration.pt"
    legacy_hash = _verified(legacy_pt, cal_meta.get("cache_sha256"))
    legacy_payload = _torch_load(legacy_pt)
    if not isinstance(legacy_payload, dict) or not {"identity", "features", "encoder", "tensor", "raw_mean", "raw_std", "feature_mean", "feature_std"} <= set(legacy_payload):
        raise ArtifactError("Unsupported legacy calibration payload format")
    _check_finite_tree({k: v for k, v in legacy_payload.items() if k != "features"}, "legacy calibration")
    expected_cal_identity = {"study_id": legacy_id, "subject": subject, "seed": seed,
        "split_id": legacy_split["split_id"], "dataset_fingerprint": legacy_split["fingerprint"]}
    if any(legacy_payload["identity"].get(k) != v for k, v in expected_cal_identity.items()):
        raise ArtifactError("Swapped task identity in legacy tensor payload")
    features = legacy_payload["features"]
    if tuple(features.shape) != (len(loaded_ids), 22, 4, 32):
        raise ArtifactError(f"Legacy feature cache has unsupported shape {tuple(features.shape)}")
    # Slice immediately. The all-partition tensor never enters a public dataclass.
    legacy_train = _array(features[list(legacy_split["train"])].detach().cpu().numpy())
    legacy_validation = _array(features[list(legacy_split["validation"])].detach().cpu().numpy())
    legacy_view = LegacyView(legacy_parts["train"], legacy_parts["validation"], legacy_train,
        legacy_validation, _array(legacy_payload["raw_mean"].detach().cpu().numpy()),
        _array(legacy_payload["raw_std"].detach().cpu().numpy()), _array(legacy_payload["feature_mean"].detach().cpu().numpy()),
        _array(legacy_payload["feature_std"].detach().cpu().numpy()), dict(legacy_payload["encoder"]),
        dict(legacy_payload["tensor"]), dict(legacy_payload.get("encoder_selection", {})),
        {"pretraining_mha_head": "unavailable_not_persisted", "downstream_heads": "unavailable_not_persisted"})

    stats = base_dataset.get("normalization_stats", {})
    mean, std = _array(stats.get("mean")), _array(stats.get("std"))
    if mean.shape != (22,) or std.shape != (22,) or (std <= 0).any():
        raise ArtifactError("Baseline normalization statistics have invalid shape or values")
    checkpoints = {}
    baseline_hashes = {}
    for arm in ("eegnet_reference__full", "masked_eegnet__full", "masked_eegnet__mixed"):
        arm_dir = base_dir / "ARMS" / arm
        record = _json(arm_dir / "result.json")
        if record.get("subject") != subject or record.get("seed") != seed or record.get("status") != "complete":
            raise ArtifactError(f"Wrong or incomplete selected checkpoint identity for {arm}")
        expected_result_identity = {"subject": subject, "seed": seed, "split_id": base_split["split_id"],
            "config_sha256": baseline_manifest["identity"].get("config_sha256"),
            "source_sha256": baseline_manifest["identity"].get("source_sha256"),
            "synthetic": False, "protocol": "within_session",
            "data_fingerprint": base_dataset.get("data_fingerprint")}
        result_identity = record.get("identity", {})
        if not isinstance(result_identity, dict) or any(result_identity.get(k) != v for k, v in expected_result_identity.items()):
            raise ArtifactError(f"Selected checkpoint result identity mismatch for {arm}")
        checkpoint_path = arm_dir / "checkpoint.pt"
        checkpoint_hash = _verified(checkpoint_path, record.get("checkpoint_sha256"))
        history_path = arm_dir / "history.json"
        history_hash = _verified(history_path, record.get("history_sha256"))
        checkpoint = _torch_load(checkpoint_path)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "agfl-baseline-checkpoint-v1" or not isinstance(checkpoint.get("state_dict"), dict):
            raise ArtifactError(f"Unsupported baseline checkpoint format for {arm}")
        _check_finite_tree(checkpoint, f"baseline checkpoint {arm}")
        history = _json(history_path)
        selected_epoch = record.get("selected_epoch")
        if type(selected_epoch) is not int or selected_epoch < 1:
            raise ArtifactError(f"Missing saved validation-selected epoch for {arm}")
        # Keep the immutable selected history row so clean checkpoint replay
        # can compare its recorded train/validation metrics against inference.
        selected = tuple(dict(row)
                         for row in history if isinstance(row, dict) and row.get("epoch") == selected_epoch)
        if len(selected) != 1:
            raise ArtifactError(f"History does not contain selected epoch {selected_epoch} for {arm}")
        checkpoints[arm] = BaselineCheckpoint(arm, selected_epoch, str(record.get("selection_policy", "")),
            checkpoint["model_type"], dict(checkpoint.get("constructor", {})), checkpoint["state_dict"],
            dict(checkpoint.get("metadata", {})), selected)
        baseline_hashes[f"{arm}/checkpoint.pt"] = checkpoint_hash
        baseline_hashes[f"{arm}/history.json"] = history_hash
    base_view = BaselineView(base_parts["train"], base_parts["validation"], mean, std, checkpoints)
    base_source_map = baseline_manifest["identity"].get("source_sha256", {})
    legacy_source_map = legacy_manifest.get("source", {}).get("files_sha256", {})
    base_source = SourceMetadata(baseline_id, digest(base_source_map), file_sha256(Path(cfg["baseline_dir"]) / "study.json"),
        str(base_dir), subject, seed, base_split["split_id"], base_dataset.get("data_fingerprint", ""),
        {**baseline_hashes, "recording": recording_hash}, _source_status("baseline", base_source_map))
    legacy_source = SourceMetadata(legacy_id, digest(legacy_source_map), file_sha256(Path(cfg["legacy_dir"]) / "study.json"),
        str(legacy_dir), subject, seed, legacy_split["split_id"], legacy_split["fingerprint"],
        {"calibration.pt": legacy_hash}, _source_status("legacy", legacy_source_map))
    return TaskSources(subject, seed, base_source, legacy_source, base_view, legacy_view)
