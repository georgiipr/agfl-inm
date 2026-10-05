"""Fixed exploratory train/validation protocol and independent audit identity."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "agfl-encoder-audit-v1"
PACKAGES = ("numpy", "scipy", "torch", "scikit-learn", "mne", "tqdm")
PATH_KEYS = ("data_dir", "baseline_dir", "legacy_dir", "output_dir")
FIXED = {
    "schema_name": SCHEMA,
    "protocol": "within_session",
    "partitions": ["train", "validation"],
    "preprocessing": {"sessions": ["T"], "channels": 22, "sampling_rate": 250,
        "windows": 4, "window_samples": 250, "trial_samples": 1000,
        "offset_seconds": 0.0, "filter_low_hz": 2.0, "filter_high_hz": 30.0,
        "filter_scope": "window", "artifact_policy": "exclude"},
    "normalization": {"baseline_raw_epsilon": 1e-8, "legacy_raw_epsilon": 1e-12,
                      "legacy_feature_epsilon": 1e-6},
    "budgets": {"cpu_probe_fits_per_task": 4, "neural_refits": 0},
    "probes": {"views": ["features_ordered", "features_window_mean", "core_ordered"],
        "negative_control": "features_ordered_shuffled", "shuffle_seed_offset": 700001,
        "scaler": "StandardScaler_train_only", "penalty": "l2", "C": 1.0,
        "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4, "random_state": 0,
        "class_weight": None, "multiclass": "multinomial", "search": False},
    "conditions": [
        {"name": "full_22", "pattern": "full", "retained": 22, "repeats": 1},
        {"name": "random_static_16", "pattern": "random_static", "retained": 16, "repeats": 5},
        {"name": "dynamic_random_16", "pattern": "dynamic_random", "retained": 16, "repeats": 5},
        {"name": "random_static_6", "pattern": "random_static", "retained": 6, "repeats": 5},
        {"name": "dynamic_random_6", "pattern": "dynamic_random", "retained": 6, "repeats": 5}],
    "tolerances": {"features": {"rtol": 1e-4, "atol": 1e-5},
        "probabilities": {"rtol": 1e-4, "atol": 1e-5},
        "normalization": {"rtol": 1e-6, "atol": 1e-8}},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _fixed(value, expected, field):
    if isinstance(expected, dict):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"{field}: expected exactly keys {sorted(expected)}; unknown settings forbidden")
        for key in expected:
            _fixed(value[key], expected[key], f"{field}.{key}")
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{field}: fixed protocol requires {expected!r}")
        for index, item in enumerate(expected):
            _fixed(value[index], item, f"{field}[{index}]")
    else:
        # Python's True == 1 must not permit Boolean counts or budgets.
        valid_type = (type(value) in (int, float) if type(expected) is float
                      else type(value) is type(expected))
        if not valid_type or value != expected:
            raise ValueError(f"{field}: fixed protocol requires {expected!r}")


def load_config(path):
    """Validate all settings; resolve paths against the config, including symlinks."""
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                     object_pairs_hook=_unique_object)
    keys = set(FIXED) | set(PATH_KEYS) | {"name", "subjects", "seeds"}
    if not isinstance(cfg, dict) or set(cfg) != keys:
        raise ValueError(f"config: expected exactly keys {sorted(keys)}; unknown selection/search settings forbidden")
    for key, expected in FIXED.items():
        _fixed(cfg[key], expected, key)
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    for field, low, high in (("subjects", 1, 9), ("seeds", 0, 2)):
        values = cfg[field]
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or not low <= v <= high for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"{field} must be unique integers in {low}..{high}")
    for key in PATH_KEYS:
        value = cfg[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path")
        cfg[key] = str((path.parent / Path(value).expanduser()).resolve())
    output = Path(cfg["output_dir"])
    if output == ROOT:
        raise ValueError("output_dir must not equal the repository root")
    for key in ("data_dir", "baseline_dir", "legacy_dir"):
        source = Path(cfg[key])
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"output_dir overlaps {key} (equal, contained, or containing)")
    if Path(cfg["baseline_dir"]) == Path(cfg["legacy_dir"]):
        raise ValueError("baseline_dir and legacy_dir must identify distinct input protocols")
    cfg["config_path"] = str(path)
    return cfg


def tasks(cfg):
    """Subject-major pairs in declared order; task zero is A01/seed 0 by default."""
    return [(subject, seed) for subject in cfg["subjects"] for seed in cfg["seeds"]]


def audit_identity(cfg):
    """Hash new source/config/packages independently of historical input identities."""
    files = sorted([*ROOT.glob("*.py"), *(ROOT / "agfl").rglob("*.py"),
                    *(ROOT / "inm").rglob("*.py")])
    source = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in files}
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity = {"schema_name": SCHEMA, "config_sha256": file_sha256(cfg["config_path"]),
        "resolved_config_sha256": digest(cfg), "source_files_sha256": source,
        "source_sha256": digest(source), "packages": packages,
        "python": platform.python_version()}
    return {**identity, "audit_id": digest(identity)}
