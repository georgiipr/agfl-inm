"""Fixed, dependency-light declaration for the encoder-candidate study."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "agfl-encoder-candidates-v1"
ARM_IDS = ("local_control", "local_power", "spatial_eegnet",
           "spatial_filterbank", "spatial_transformer")
PACKAGE_NAMES = ("numpy", "scipy", "torch", "scikit-learn", "mne", "tqdm")
PATH_KEYS = ("data_dir", "split_source_dir", "output_dir")

FIXED = {
    "schema_name": SCHEMA,
    "partitions": ["train", "validation"],
    "training_regime": "full",
    "preprocessing": {"sessions": ["T"], "channels": 22,
        "channel_order": "canonical_22", "sampling_rate": 250,
        "windows": 4, "window_samples": 250, "trial_samples": 1000,
        "cue_offset_seconds": 0.0, "filter_low_hz": 2.0,
        "filter_high_hz": 30.0, "filter_scope": "window",
        "artifact_policy": "exclude", "augmentation": "none",
        "normalization": "training_channel_mean_population_std_floor_1e-8"},
    "arms": [
        {"id": "local_control", "encoder": "EEGWindowEncoder",
         "encoder_settings": {"f1": 16, "d": 2, "f2": 32,
             "kernel_length": 32, "pools": [8, 16], "dropout": 0.5},
         "head": "FeatureClassifier", "head_settings": {"representation": "baseline",
             "attention": "MHA", "dim": 32, "heads": 4, "depth": 1,
             "dropout": 0.1}},
        {"id": "local_power", "encoder": "LocalPowerEncoder",
         "encoder_settings": {"kernels": [16, 32, 64, 128], "filters_per_branch": 8,
             "input_channels": 1, "stride": 1, "padding": "explicit_zero_same",
             "bias": False, "variance": "population_over_250_responses",
             "log_epsilon": 1e-6, "features": 32, "layout": "B22W4F32"},
         "head": "FeatureClassifier", "head_settings": {"representation": "baseline",
             "attention": "MHA", "dim": 32, "heads": 4, "depth": 1,
             "dropout": 0.1}},
        {"id": "spatial_eegnet", "encoder": "EEGNetClassifier",
         "encoder_settings": {"f1": 8, "depth_multiplier": 2, "f2": 16,
             "temporal_kernel": 64, "separable_kernel": 16,
             "pooling": [4, 8], "dropout": 0.5, "head_type": "flatten",
             "mask_conditioned": True}, "head": "EEGNet flatten"},
        {"id": "spatial_filterbank", "encoder": "SpatialFilterBank",
         "encoder_settings": {"bands_hz": [[4, 8], [8, 12], [12, 16],
             [16, 20], [20, 24], [24, 30]], "fir_taps": 81,
             "fir_window": "hamming", "fir_design": "scipy.signal.firwin",
             "sampling_rate": 250, "pass_zero": False,
             "spatial_filters_per_band": 4, "spatial_filter_max_norm": 1.0,
             "variance": "population_over_250_responses", "log_epsilon": 1e-6,
             "dropout": 0.5, "availability_flags": 88,
             "classifier": "linear_four_class"}, "head": "linear"},
        {"id": "spatial_transformer", "encoder": "EEGNetTransformer",
         "encoder_settings": {"front_end": "spatial_eegnet_without_classifier",
             "front_end_features": 16, "temporal_positions": 31,
             "d_model": 32, "position_encoding": "fixed_sinusoidal",
             "layers": 1, "heads": 4, "dim_feedforward": 64,
             "activation": "gelu", "dropout": 0.1, "norm_first": True,
             "batch_first": True, "pooling": "mean_positions",
             "availability_flags": 88, "classifier": "linear_four_class"},
         "head": "compact_transformer"}],
    "training": {"loss": "cross_entropy_unweighted", "optimizer": "AdamW",
        "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 32,
        "maximum_epochs": 250, "minimum_epochs": 75, "patience": 50,
        "gradient_clip_norm": 1.0, "warmup_epochs": 10,
        "schedule": "linear_warmup_cosine_to_zero", "seed_order": "independent_model_seed"},
    "probes": {"local_arms": ["local_control", "local_power"],
        "per_arm": ["ordered_feature_logistic", "shuffled_training_label_control"],
        "scaler": "StandardScaler_train_only", "penalty": "l2", "C": 1.0,
        "solver": "lbfgs", "max_iter": 1000, "tol": 0.0001,
        "random_state": 0, "class_weight": None, "label_shuffle_seed_offset": 700001,
        "search": False, "fits_per_task": 4},
    "conditions": [
        {"name": "full_22", "pattern": "full", "retained": 22, "repeats": 1},
        {"name": "random_static_16", "pattern": "random_static", "retained": 16, "repeats": 5},
        {"name": "dynamic_random_16", "pattern": "dynamic_random", "retained": 16, "repeats": 5},
        {"name": "random_static_6", "pattern": "random_static", "retained": 6, "repeats": 5},
        {"name": "dynamic_random_6", "pattern": "dynamic_random", "retained": 6, "repeats": 5}],
    "selection": {"metric": "validation_balanced_accuracy", "tie_breaker": "validation_log_loss",
        "tie_breaker_2": "earlier_epoch", "input": "full_22_only",
        "test_selection": False},
    "comparisons": {"primary": ["local_power", "local_control"],
        "secondary": [["spatial_filterbank", "spatial_eegnet"],
                      ["spatial_transformer", "spatial_eegnet"]],
        "screening_mean_gain": 0.02, "screening_positive_participants": 6,
        "bootstrap_repeats": 2000, "bootstrap_seed": 20261003},
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
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


def _same_fixed(value, expected, field):
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


def load_config(path, *, allow_existing_output: bool = False) -> dict:
    """Validate the exact declaration; execution may inspect an occupied output.

    This option only permits read-only config validation when the declared output
    is already a directory. Task and fit writers remain responsible for strict
    provenance, completeness, and checksum checks before reuse.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read candidate config {config_path}: {error}") from error
    required = set(FIXED) | set(PATH_KEYS) | {"name", "subjects", "seeds", "synthetic"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"config: expected exactly keys {sorted(required)}; unknown settings/search forbidden")
    if type(cfg["synthetic"]) is not bool:
        raise ValueError("synthetic must be Boolean")
    for key, expected in FIXED.items():
        if key == "training" and cfg["synthetic"]:
            training = cfg[key]
            if not isinstance(training, dict) or set(training) != set(expected):
                raise ValueError("training: synthetic fixture must keep the fixed setting keys")
            for budget in ("maximum_epochs", "minimum_epochs", "patience", "batch_size",
                           "warmup_epochs"):
                value = training[budget]
                if type(value) is not int or value < (0 if budget in ("patience", "warmup_epochs") else 1) or value > expected[budget]:
                    raise ValueError(f"training.{budget} may only be reduced in a synthetic fixture")
            if training["minimum_epochs"] > training["maximum_epochs"]:
                raise ValueError("synthetic training minimum_epochs must not exceed maximum_epochs")
            if training["warmup_epochs"] >= training["maximum_epochs"]:
                raise ValueError("synthetic training warmup_epochs must be below maximum_epochs")
            normalized_training = dict(training)
            for budget in ("maximum_epochs", "minimum_epochs", "patience", "batch_size",
                           "warmup_epochs"):
                normalized_training[budget] = expected[budget]
            _same_fixed(normalized_training, expected, key)
        else:
            _same_fixed(cfg[key], expected, key)
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    for field, low, high, exact in (("subjects", 1, 9, list(range(1, 10))),
                                    ("seeds", 0, 2**31 - 1, [0, 1, 2])):
        values = cfg[field]
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or not low <= v <= high for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"{field} must contain unique integers in {low}..{high}")
        allowed = (set(values).issubset(exact) if cfg["synthetic"] else values == exact)
        if not allowed:
            raise ValueError(f"{field} must equal the fixed cohort {exact!r}; only synthetic fixtures may shrink it")
    for key in PATH_KEYS:
        value = cfg[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path")
        cfg[key] = str((config_path.parent / Path(value).expanduser()).resolve())
    output = Path(cfg["output_dir"])
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output_dir must not equal or contain the repository root")
    for key in ("data_dir", "split_source_dir"):
        source = Path(cfg[key])
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"output_dir overlaps {key} (equal, contained, or containing)")
    if Path(cfg["data_dir"]) == Path(cfg["split_source_dir"]):
        raise ValueError("data_dir and split_source_dir must identify distinct inputs")
    if output.exists() and (not output.is_dir() or
                            (any(output.iterdir()) and not allow_existing_output)):
        raise ValueError("output_dir already contains artifacts or is not a directory; use a fresh output identity")
    cfg["config_path"] = str(config_path)
    return cfg


def tasks(cfg) -> list[dict]:
    """Enumerate declared subject-major tasks."""
    return [{"subject": subject, "seed": seed}
            for subject in cfg["subjects"] for seed in cfg["seeds"]]


def arms(cfg) -> list[str]:
    """Return the five fixed arm identifiers in protocol order."""
    return [arm["id"] for arm in cfg["arms"]]


def _source_hashes() -> dict[str, str]:
    paths = {*ROOT.glob("*.py"), *(ROOT / "agfl").rglob("*.py"),
             *(ROOT / "inm").rglob("*.py")}
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def study_identity(cfg) -> dict:
    """Return reproducibility identity, independent of historical checkpoints."""
    source_files = _source_hashes()
    identity = {"schema_name": SCHEMA, "config_sha256": file_sha256(cfg["config_path"]),
        "resolved_config_sha256": digest({key: value for key, value in cfg.items()
                                           if key != "config_path"}),
        "source_files_sha256": source_files, "source_sha256": digest(source_files),
        "packages": _package_versions(), "python": platform.python_version()}
    return {**identity, "study_id": digest(identity)}


def inspect_inputs(cfg) -> dict:
    """Inventory expected inputs without importing numeric modules or loading EEG."""
    data_dir, split_dir = Path(cfg["data_dir"]), Path(cfg["split_source_dir"])
    manifest_candidates = [split_dir / name for name in ("study.json", "manifest.json",
                                                          "study_manifest.json")]
    available_manifests = [str(path) for path in manifest_candidates if path.is_file()]
    recordings = []
    for subject in range(1, 10):
        path = data_dir / f"A{subject:02d}T.gdf"
        recordings.append({"subject": subject, "path": str(path), "available": path.is_file()})
    split_entries = []
    for task in tasks(cfg):
        subject, seed = task["subject"], task["seed"]
        directory = split_dir / "artifacts" / f"A{subject:02d}_seed_{seed}"
        required_paths = [directory / "split.json", directory / "dataset.json"]
        split_entries.append({"subject": subject, "seed": seed,
            "directory": str(directory), "split_json": required_paths[0].is_file(),
            "dataset_json": required_paths[1].is_file(),
            "manifest_candidates": available_manifests})
    versions = _package_versions()
    required_packages = ("numpy", "scipy", "torch", "scikit-learn", "mne")
    missing_packages = [name for name in required_packages if versions[name] is None]
    missing_recordings = [entry["path"] for entry in recordings if not entry["available"]]
    missing_splits = [entry for entry in split_entries
                      if not entry["split_json"] or not entry["dataset_json"]]
    return {"status": "ready" if not missing_packages and not missing_recordings and
            not missing_splits and available_manifests else "unavailable",
        "recordings": recordings, "splits": split_entries, "packages": versions,
        "missing_packages": missing_packages, "missing_recordings": missing_recordings,
        "missing_split_metadata": missing_splits,
        "study_manifest": {"candidates": [str(path) for path in manifest_candidates],
                           "available": available_manifests},
        "historical_checkpoint_source_mismatch": "not an input; fresh fits use current source",
        "limits": ["inventory only: EEG arrays and checkpoints were not loaded",
                   "metadata contents and recording hashes are verified by the data session"]}
