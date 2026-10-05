"""Fixed task-driven completion declaration and read-only provenance.

Planning, identity and input inventory use the standard library only. Presence
and byte hashes do not establish historical replay compatibility or readiness.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

from inm.completion_transformer.protocol import inspect_inputs as _historical_inventory

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "agfl-task-driven-completion-v1"
ARM_IDS = ("spatial_transformer",)
HISTORICAL_ARM_IDS = ("spatial_eegnet", "spatial_transformer")
STRATEGY_IDS = ("zero", "covariance_frozen", "covariance_learned", "tucker_frozen", "tucker_learned")
LEARNED_STRATEGY_IDS = ("covariance_learned", "tucker_learned")
CONDITIONS = (
    {"name": "full_22", "pattern": "full", "retained": 22, "repeats": 1},
    {"name": "random_static_16", "pattern": "random_static", "retained": 16, "repeats": 5},
    {"name": "dynamic_random_16", "pattern": "dynamic_random", "retained": 16, "repeats": 5},
    {"name": "random_static_6", "pattern": "random_static", "retained": 6, "repeats": 5},
    {"name": "dynamic_random_6", "pattern": "dynamic_random", "retained": 6, "repeats": 5},
)
PATH_KEYS = ("data_dir", "candidate_source_dir", "split_source_dir", "output_dir")
PACKAGE_NAMES = ("numpy", "scipy", "torch", "scikit-learn", "mne")
FIXED = {
    "schema_name": SCHEMA,
    "protocol": {
        "partitions": ["train", "validation"], "sessions": ["T"],
        "channels": 22, "channel_order": "canonical_22", "windows": 4,
        "window_samples": 250, "trial_samples": 1000, "sampling_rate": 250,
        "filter_hz": [2.0, 30.0],
        "normalization": "historical_training_channel_mean_population_std_floor_1e-8",
        "backbones": list(ARM_IDS), "strategies": list(STRATEGY_IDS),
        "conditions": [dict(row) for row in CONDITIONS],
        "execution": {"device": "cpu", "threads": 1, "cuda_verified": False,
            "restore_python_numpy_torch_rng_and_threads": True,
            "independent_scoped_model_data_rng": True},
        "historical": {
            "verification": "inm.completion_transformer.replay.verify_historical_task",
            "replay": "inm.completion_transformer.replay.replay_full_input",
            "verified_backbones": list(HISTORICAL_ARM_IDS),
            "original_full_input_prediction_replay_required": True,
            "read_only": True, "backbone_fits": 0,
            "backbone_checkpoints": "historical_full_input_validation_selected",
            "frozen_weights_batchnorm_dropout": True, "input_gradients": True,
            "whole_trial_historical_eegnet_convolutions": True},
        "covariance": {
            "estimator": "training_channel_second_moment", "fit_partitions": ["train"],
            "initialization_jitter": 1e-6, "parameterization": "cholesky_softplus_diagonal",
            "diagonal_floor": 1e-6, "inverse_softplus_initialization": True,
            "ridge_scale": 0.001, "ridge_multiplier": "mean_diagonal",
            "paired_initialization_and_inference": True,
            "learned_parameters": "free_lower_triangle",
            "anchor_parameters": "free_cholesky_parameters"},
        "tucker": {
            "implementation": "inm.tensor_attention.Tucker2", "fit_partitions": ["train"],
            "rank_channels": 4, "rank_features": 16, "ridge": 0.001,
            "ridge_penalty": "n_observed*250*0.001",
            "solver": "differentiable_observed_entry_ridge_float64_both_arms",
            "epochs": 30, "learning_rate": 0.01, "batch_size": 64,
            "seed_offset": 810001, "restore_ambient_rng": True,
            "paired_initialization_and_inference": True,
            "unit_columns_after_optimizer_step": True, "anchor_parameters": "U_and_V"},
        "training": {
            "strategies": list(LEARNED_STRATEGY_IDS), "partitions": ["train"],
            "classification_loss": "cross_entropy_through_frozen_backbone",
            "reconstruction_loss": "mean_squared_error_artificially_hidden_training_samples",
            "reconstruction_weight": 0.1,
            "anchor_loss": "mean_squared_parameter_displacement_from_initial_state",
            "anchor_weight": 1e-4, "empty_reconstruction_target": "differentiable_zero",
            "skip_wholly_full_batch_updates": True,
            "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 32,
            "maximum_epochs": 100, "minimum_epochs": 10, "patience": 20,
            "gradient_clip": 1.0, "scheduler": None, "weight_decay": 0.0,
            "paired_batch_order_and_masks": True},
        "training_masks": {
            "retained": [22, 16, 6], "retained_distribution": "uniform_per_example",
            "nonfull_patterns": ["random_static", "dynamic_random"],
            "nonfull_pattern_distribution": "uniform",
            "implementation": "make_mask_bank", "partition": "train",
            "sample_ids_required": True, "seed_derivation": "epoch_and_repeat",
            "disjoint_from_validation": True},
        "selection": {
            "partition": "validation", "metric": "mean_degraded_balanced_accuracy",
            "conditions": [row["name"] for row in CONDITIONS[1:]], "repeats": 5,
            "tie_break": ["lower_mean_degraded_log_loss", "earlier_epoch"],
            "epoch_zero_eligible": True, "minimum_training_epochs": 10,
            "frozen_controls_supervised_selection": False, "test_data": False,
            "full_input_accuracy_gain_objective": False},
        "mask_policy": {
            "preserve_original_boolean_flags": True,
            "observed_values_selected_before_arithmetic": True,
            "hidden_nan_inf_invariance": True, "reject_observed_nonfinite": True,
            "observed_samples_bitwise_unchanged": True, "all_missing_window": False,
            "full_input_bypasses_completion": True, "complete_independently_per_window": True},
        "artifacts": {
            "numeric_probabilities_labels_sample_ids": True, "masks_and_hashes": True,
            "initial_and_selected_completion_states": True, "histories_selected_epoch": True,
            "checksums_and_independent_replay": True, "resume_verified_complete_only": True,
            "recheck_historical_input_hashes": True, "synthetic_excluded_from_real_reports": True},
        "evaluation": {
            "partition": "validation", "metrics": ["balanced_accuracy", "log_loss"],
            "paired_masks_across_strategies": True, "selection_masks_reused": True,
            "repeat_then_seed_then_participant_equal_averaging": True,
            "primary": "mean_degraded_tucker_learned_minus_tucker_frozen",
            "secondary": ["mean_degraded_tucker_learned_minus_covariance_frozen",
                "mean_degraded_tucker_learned_minus_covariance_learned",
                "mean_degraded_covariance_learned_minus_covariance_frozen",
                "mean_degraded_tucker_learned_minus_zero"],
            "screening_mean_gain": 0.02, "screening_positive_participants": 6,
            "screening_separate_from_strong_control_comparisons": True,
            "bootstrap_unit": "paired_participant", "bootstrap_repeats": 2000,
            "bootstrap_seed": 20261005, "intervals": "exploratory_unadjusted",
            "suppress_incomplete_real_cohort_means": True,
            "preserve_negative_effects_and_per_condition_log_loss": True,
            "independent_confirmation": False},
    },
}


def digest(value: Any) -> str:
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


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def validate_output_path(cfg: dict) -> Path:
    """Recheck resolved output isolation immediately before any future write.

    This does not authorize resume or verify artifacts. Execution must also check
    persisted study identity and completed-task integrity before reusing output.
    """
    output = Path(cfg["output_dir"]).resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output_dir must not equal or contain the repository root")
    for key in PATH_KEYS[:-1]:
        if _overlaps(output, Path(cfg[key]).resolve()):
            raise ValueError(f"output_dir overlaps {key}")
    if "config_path" in cfg and _overlaps(output, Path(cfg["config_path"]).resolve()):
        raise ValueError("output_dir overlaps config_path")
    # All repository entries except the results container are protected. This
    # includes untracked source, documentation, workflows and hidden state.
    for path in ROOT.iterdir():
        if path.name != "results" and _overlaps(output, path.resolve()):
            raise ValueError(f"output_dir overlaps protected repository path {path.name}")
    result_root = ROOT / "results"
    if output == result_root.resolve():
        raise ValueError("output_dir must not be the results container")
    if result_root.is_dir():
        for path in result_root.iterdir():
            own_root = path.name.startswith("task-driven-completion-") and output == path.resolve()
            if not own_root and _overlaps(output, path.resolve()):
                raise ValueError(f"output_dir overlaps historical results {path.name}")
    if output.is_dir():
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"output_dir contains a symlink descendant: {path}")
    return output


def load_config(path: str | Path, *, allow_existing_output: bool = False) -> dict:
    """Validate exact science and resolve paths by config location.

    ``allow_existing_output`` permits read-only inspection, never resume by
    itself. A future execution layer must verify identity and complete artifacts.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read task-driven-completion config {config_path}: {error}") from error
    required = set(FIXED) | set(PATH_KEYS) | {"name", "subjects", "seeds", "synthetic"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"config: expected exactly keys {sorted(required)}; unknown settings forbidden")
    for key, expected in FIXED.items():
        _same_fixed(cfg[key], expected, key)
    if type(cfg["synthetic"]) is not bool:
        raise ValueError("synthetic must be Boolean")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    for field, exact in (("subjects", list(range(1, 10))), ("seeds", [0, 1, 2])):
        values = cfg[field]
        if (not isinstance(values, list) or not values or
                any(type(value) is not int or value not in exact for value in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"{field} must contain unique integers from {exact}")
        if not cfg["synthetic"] and values != exact:
            raise ValueError(f"{field} must equal the fixed cohort {exact}")
    for key in PATH_KEYS:
        value = cfg[key]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{key} must be a nonempty path without NUL")
        cfg[key] = str((config_path.parent / Path(value).expanduser()).resolve())
    if len({cfg[key] for key in PATH_KEYS[:-1]}) != len(PATH_KEYS[:-1]):
        raise ValueError("data and historical input paths must be distinct")
    cfg["config_path"] = str(config_path)
    output = validate_output_path(cfg)
    if output.exists() and (not output.is_dir() or
                            (any(output.iterdir()) and not allow_existing_output)):
        raise ValueError("output_dir already contains artifacts or is not a directory; use a fresh output identity")
    return cfg


def tasks(cfg: dict) -> list[dict[str, int]]:
    return [{"subject": subject, "seed": seed}
            for subject in cfg["subjects"] for seed in cfg["seeds"]]


def plan(cfg: dict) -> dict:
    planned = tasks(cfg)
    return {"status": "planned", "tasks": len(planned), "task_rows": planned,
        "backbones": list(ARM_IDS), "strategies": list(STRATEGY_IDS),
        "strategies_per_task": len(STRATEGY_IDS),
        "supervised_completion_fits": len(planned) * len(LEARNED_STRATEGY_IDS),
        "backbone_fits": 0, "tucker_initializations": len(planned),
        "covariance_initializations": len(planned),
        "evaluation_rows_per_strategy": sum(row["repeats"] for row in CONDITIONS),
        "partitions": ["train", "validation"], "device": "cpu", "threads": 1,
        "real_fitting_ready": False,
        "limitations": ["Plan only; no historical verification, numerical replay or fitting."]}


def _source_hashes() -> dict[str, str]:
    paths = {*ROOT.glob("*.py"), *(ROOT / "agfl").rglob("*.py"), *(ROOT / "inm").rglob("*.py")}
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


def _historical_manifest_identity(cfg: dict) -> dict[str, str | None]:
    candidate, split = Path(cfg["candidate_source_dir"]), Path(cfg["split_source_dir"])
    paths = [candidate / "report/evidence.json", split / "study.json",
             ROOT / "configs/encoder-candidates-reproducible.json"]
    for row in tasks(cfg):
        name = f"A{row['subject']:02d}_seed_{row['seed']}"
        paths.extend(candidate / "tasks" / name / item for item in ("task.json", "data_provenance.json"))
        paths.extend(split / "artifacts" / name / item for item in ("split.json", "dataset.json"))
    return {str(path): file_sha256(path) if path.is_file() else None for path in paths}


def study_identity(cfg: dict) -> dict:
    """Hash config, reused/new Python source, packages and historical metadata.

    Missing historical manifests remain explicit nulls, never a readiness claim.
    Task execution must additionally recheck all historical artifact checksums.
    """
    source_files = _source_hashes()
    identity = {"schema_name": SCHEMA,
        "config_sha256": file_sha256(cfg["config_path"]),
        "resolved_config_sha256": digest({key: value for key, value in cfg.items() if key != "config_path"}),
        "source_files_sha256": source_files, "source_sha256": digest(source_files),
        "historical_manifest_sha256": _historical_manifest_identity(cfg),
        "packages": _package_versions(), "python": platform.python_version()}
    return {**identity, "study_id": digest(identity)}


def inspect_inputs(cfg: dict) -> dict:
    """Read-only file inventory via the predecessor; no array deserialization.

    The strict historical verifier also reads NumPy prediction arrays, and is
    deliberately deferred to execution so this command remains dependency-light.
    """
    inventory = _historical_inventory(cfg)
    return {**inventory, "identity": study_identity(cfg),
        "device": "cpu", "threads": 1, "cuda_verified": False,
        "real_fitting_ready": False, "historical_verification": "not_run",
        "original_full_input_replay": "not_run",
        "required_execution_gates": [
            "inm.completion_transformer.replay.verify_historical_task (both historical backbones)",
            "inm.completion_transformer.replay.replay_full_input (original predictions)",
            "independently accepted synthetic implementation and explicit real pilot operation"],
        "limitations": inventory["limitations"] + [
            "Inventory success is not fitting readiness; strict verification and original replay remain required.",
            "CPU one thread only; CUDA is unverified.",
            "No test arrays, test labels or test scores are loaded; no output is created."]}
