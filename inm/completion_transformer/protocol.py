"""Frozen protocol, identity and metadata-only input inventory.

This module deliberately stays within the Python standard library. In
particular, plan and preflight commands must not import torch, load EEG arrays,
or deserialize checkpoints.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "agfl-completion-transformer-v1"
ARM_IDS = ("spatial_eegnet", "spatial_transformer")
STRATEGY_IDS = ("zero", "covariance", "tucker")
CONDITIONS = (
    {"name": "full_22", "pattern": "full", "retained": 22, "repeats": 1},
    {"name": "random_static_16", "pattern": "random_static", "retained": 16, "repeats": 5},
    {"name": "dynamic_random_16", "pattern": "dynamic_random", "retained": 16, "repeats": 5},
    {"name": "random_static_6", "pattern": "random_static", "retained": 6, "repeats": 5},
    {"name": "dynamic_random_6", "pattern": "dynamic_random", "retained": 6, "repeats": 5},
)
PACKAGE_NAMES = ("numpy", "scipy", "torch", "scikit-learn", "mne")
PATH_KEYS = ("data_dir", "candidate_source_dir", "split_source_dir", "output_dir")

FIXED: dict[str, Any] = {
    "schema_name": SCHEMA,
    "protocol": {
        "partitions": ["train", "validation"], "sessions": ["T"],
        "channels": 22, "channel_order": "canonical_22", "windows": 4,
        "window_samples": 250, "trial_samples": 1000, "sampling_rate": 250,
        "filter_hz": [2.0, 30.0],
        "normalization": "historical_training_channel_mean_population_std_floor_1e-8",
        "backbones": list(ARM_IDS), "strategies": list(STRATEGY_IDS),
        "conditions": [dict(row) for row in CONDITIONS],
        "covariance": {"estimator": "training_channel_second_moment",
            "ridge_scale": 0.001, "ridge_multiplier": "mean_diagonal", "mask_tuning": False},
        "tucker": {"implementation": "inm.tensor_attention.Tucker2",
            "fit_partitions": ["train"], "rank_channels": 4, "rank_features": 16,
            "ridge": 0.001, "epochs": 30, "learning_rate": 0.01,
            "batch_size": 64, "seed_offset": 810001, "restore_ambient_rng": True},
        "selection": {"checkpoints": "historical_full_input_validation_selected",
            "reselect": False, "test_data": False},
        "mask_policy": {"preserve_original_boolean_flags": True,
            "observed_values_selected_before_arithmetic": True,
            "all_missing_window": False, "full_input_bypasses_completion": True},
        "evaluation": {"metrics": ["balanced_accuracy", "log_loss"],
            "repeat_then_seed_then_participant_equal_averaging": True,
            "primary": "mean_degraded_tucker_minus_zero_spatial_transformer",
            "secondary": ["mean_degraded_tucker_minus_zero_spatial_eegnet",
                "mean_degraded_tucker_minus_covariance_spatial_transformer",
                "difference_in_differences_tucker_benefit_transformer_minus_eegnet"],
            "screening_mean_gain": 0.02, "screening_positive_participants": 6,
            "bootstrap_repeats": 2000, "bootstrap_seed": 20261005},
    },
}


def digest(value: Any) -> str:
    """Hash a JSON-compatible value with stable key ordering."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _same_fixed(value: Any, expected: Any, field: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"{field}: expected exactly keys {sorted(expected)}; unknown settings forbidden")
        for key, item in expected.items():
            _same_fixed(value[key], item, f"{field}.{key}")
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{field}: fixed protocol requires {expected!r}")
        for index, item in enumerate(expected):
            _same_fixed(value[index], item, f"{field}[{index}]")
    elif type(value) is not type(expected) or value != expected:
        raise ValueError(f"{field}: fixed protocol requires {expected!r}")


def load_config(path: str | Path, *, allow_existing_output: bool = False) -> dict:
    """Load and validate the fixed declaration; resolve all paths by config location."""
    config_path = Path(path).expanduser().resolve()
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read completion-transformer config {config_path}: {error}") from error
    required = set(FIXED) | set(PATH_KEYS) | {"name", "subjects", "seeds", "synthetic"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"config: expected exactly keys {sorted(required)}; unknown settings forbidden")
    if type(cfg["synthetic"]) is not bool:
        raise ValueError("synthetic must be Boolean")
    for key, expected in FIXED.items():
        _same_fixed(cfg[key], expected, key)
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    for field, low, high, exact in (("subjects", 1, 9, list(range(1, 10))),
                                    ("seeds", 0, 2**31 - 1, [0, 1, 2])):
        values = cfg[field]
        if (not isinstance(values, list) or not values or
                any(type(value) is not int or not low <= value <= high for value in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"{field} must contain unique integers in {low}..{high}")
        if cfg["synthetic"]:
            if not set(values).issubset(exact):
                raise ValueError(f"synthetic {field} must be a subset of the fixed cohort {exact!r}")
        elif values != exact:
            raise ValueError(f"{field} must equal the fixed cohort {exact!r}")
    for key in PATH_KEYS:
        value = cfg[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path")
        cfg[key] = str((config_path.parent / Path(value).expanduser()).resolve())
    output = Path(cfg["output_dir"])
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output_dir must not equal or contain the repository root")
    for key in PATH_KEYS[:-1]:
        source = Path(cfg[key])
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"output_dir overlaps {key} (equal, contained, or containing)")
    if len({cfg[key] for key in PATH_KEYS[:-1]}) != len(PATH_KEYS[:-1]):
        raise ValueError("data and historical input paths must be distinct")
    if output.exists() and (not output.is_dir() or
                            (any(output.iterdir()) and not allow_existing_output)):
        raise ValueError("output_dir already contains artifacts or is not a directory; use a fresh output identity")
    cfg["config_path"] = str(config_path)
    return cfg


def tasks(cfg: dict) -> list[dict[str, int]]:
    """Return deterministic subject-major task records."""
    return [{"subject": subject, "seed": seed}
            for subject in cfg["subjects"] for seed in cfg["seeds"]]


def _source_hashes() -> dict[str, str]:
    paths = {*ROOT.glob("*.py"), *(ROOT / "agfl").rglob("*.py"),
             *(ROOT / "inm").rglob("*.py")}
    return {path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in sorted(paths) if path.is_file()}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _historical_manifest_identity(cfg: dict) -> dict[str, dict[str, str | None]]:
    """Hash the declared metadata manifests without opening tensor/checkpoint files."""
    candidates = {
        "candidate": [Path(cfg["candidate_source_dir"]) / "report" / "evidence.json"],
        "split_source": [Path(cfg["split_source_dir"]) / "study.json"],
    }
    result = {}
    for source, paths in candidates.items():
        result[source] = {path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT)
                          else str(path): file_sha256(path) if path.is_file() else None
                          for path in paths}
    return result


def study_identity(cfg: dict) -> dict:
    """Describe this replay's exact config, current source and historical manifests."""
    source_files = _source_hashes()
    historical = _historical_manifest_identity(cfg)
    identity = {
        "schema_name": SCHEMA,
        "config_sha256": file_sha256(cfg["config_path"]),
        "resolved_config_sha256": digest({key: value for key, value in cfg.items()
                                           if key != "config_path"}),
        "source_files_sha256": source_files,
        "source_sha256": digest(source_files),
        "historical_manifest_sha256": historical,
        "packages": _package_versions(),
        "python": platform.python_version(),
    }
    return {**identity, "study_id": digest(identity)}


def _inventory_file(path: Path) -> dict[str, Any]:
    """Return file presence and byte checksum only; never deserialize artifacts."""
    return {"path": str(path), "available": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None}


def _candidate_task_input(source_dir: Path, task: dict[str, int]) -> dict[str, Any]:
    subject, seed = task["subject"], task["seed"]
    task_name = f"A{subject:02d}_seed_{seed}"
    directory = source_dir / "tasks" / task_name
    files = {"task": directory / "task.json",
             "data_provenance": directory / "data_provenance.json"}
    for arm in ARM_IDS:
        model_dir = directory / "ARMS" / arm
        for artifact in ("checkpoint.pt", "result.json", "predictions.npz", "history.json"):
            files[f"{arm}_{artifact.removesuffix('.json').removesuffix('.npz').removesuffix('.pt')}"] = model_dir / artifact
    return {"subject": subject, "seed": seed, "directory": str(directory),
            "files": {name: _inventory_file(path) for name, path in files.items()}}


def _split_metadata_input(source_dir: Path, task: dict[str, int]) -> dict[str, Any]:
    subject, seed = task["subject"], task["seed"]
    directory = source_dir / "artifacts" / f"A{subject:02d}_seed_{seed}"
    files = {name: _inventory_file(directory / name) for name in ("split.json", "dataset.json")}
    return {"subject": subject, "seed": seed, "directory": str(directory), "files": files}


def inspect_inputs(cfg: dict) -> dict:
    """Inventory recordings and historical artifacts through paths and hashes only."""
    data_dir = Path(cfg["data_dir"])
    candidate_dir = Path(cfg["candidate_source_dir"])
    split_dir = Path(cfg["split_source_dir"])
    recordings = [_inventory_file(data_dir / f"A{subject:02d}T.gdf")
                  | {"subject": subject}
                  for subject in range(1, 10)]
    task_rows = tasks(cfg)
    candidate_tasks = [_candidate_task_input(candidate_dir, task) for task in task_rows]
    split_tasks = [_split_metadata_input(split_dir, task) for task in task_rows]
    manifests = {
        "candidate_evidence": _inventory_file(candidate_dir / "report" / "evidence.json"),
        "split_source_study": _inventory_file(split_dir / "study.json"),
    }
    package_versions = _package_versions()
    required_packages = ("numpy", "scipy", "torch", "scikit-learn", "mne")
    missing_packages = [name for name in required_packages if package_versions[name] is None]
    def absent(task_rows):
        return [{"subject": row["subject"], "seed": row["seed"], "files": names}
                for row in task_rows
                if (names := [name for name, item in row["files"].items()
                              if not item["available"]])]
    missing_candidate = absent(candidate_tasks)
    missing_splits = absent(split_tasks)
    missing_recordings = [row["path"] for row in recordings if not row["available"]]
    missing_manifests = [name for name, row in manifests.items() if not row["available"]]
    ready = not (missing_packages or missing_candidate or missing_splits or
                 missing_recordings or missing_manifests)
    return {
        "status": "inventory_complete" if ready else "unavailable",
        "recordings": recordings,
        "candidate_tasks": candidate_tasks,
        "split_metadata_tasks": split_tasks,
        "historical_manifests": manifests,
        "packages": package_versions,
        "missing_packages": missing_packages,
        "missing_recordings": missing_recordings,
        "missing_candidate_artifacts": missing_candidate,
        "missing_split_metadata": missing_splits,
        "missing_historical_manifests": missing_manifests,
        "limitations": [
            "Metadata-only inventory: checkpoint tensors, EEG arrays and prediction arrays were not loaded.",
            "Byte checksums establish file identity, not semantic compatibility or numerical correctness.",
            "Historical per-task identities and source/package compatibility require replay-session verification.",
        ],
    }
