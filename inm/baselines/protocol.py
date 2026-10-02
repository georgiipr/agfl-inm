"""Validated, dependency-light configuration for the baseline accuracy study."""
from __future__ import annotations

import json
import math
from pathlib import Path


SCHEMA = "agfl-baselines-v1"
PROTOCOLS = {
    "within_session": {"train": 0.6, "validation": 0.2, "test": 0.2,
                       "description": "stratified T-session 60/20/20"},
    "cross_session": {"train": 0.8, "validation": 0.2, "test": 0.0,
                      "description": "T-session 80/20; E-session exclusively test"},
}
# Evaluation includes held-out channel counts and spatial losses. The smaller
# five-bank robustness selection set in CONTRACT.md is validation-only.
_NEURAL_COVERAGE = ["full"] + [
    f"{pattern}_{retained}"
    for retained in (16, 11, 6)
    for pattern in ("random_static", "spatial_static", "dynamic_random", "dynamic_spatial")
]
_ARM_DEFINITIONS = (
    ("eegnet_reference", "full", "neural", _NEURAL_COVERAGE),
    ("masked_eegnet", "full", "neural", _NEURAL_COVERAGE),
    ("masked_eegnet", "mixed", "neural", _NEURAL_COVERAGE),
    ("covariance", "full", "classical", ["full"]),
)


def _object(value, name, required):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing, extra = set(required) - set(value), set(value) - set(required)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise ValueError(f"{name}: {'; '.join(details)}")
    return value


def _integer(value, name, low=1, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        bounds = f"{low}..{high}" if high is not None else f">={low}"
        raise ValueError(f"{name} must be an integer in {bounds}")


def _number(value, name, low=None, high=None, low_inclusive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and (value < low if low_inclusive else value <= low):
        raise ValueError(f"{name} must be {'at least' if low_inclusive else 'greater than'} {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be at most {high}")


def _resolve_path(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return str((base / path if not path.is_absolute() else path).resolve())


def load_config(path) -> dict:
    """Read and validate a config; relative paths are anchored at its directory."""
    config_path = Path(path).expanduser().resolve()
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read baseline config {config_path}: {error}") from error
    keys = ("schema_name", "name", "subjects", "seeds", "data_dir", "output_dir",
            "protocol", "split", "preprocessing", "model", "training", "selection",
            "arms", "mask_repeats", "bootstrap_repeats", "external_labels_dir")
    optional_keys = {"execution_arms", "head_ablation"}
    if not isinstance(cfg, dict) or set(cfg) - set(keys) - optional_keys or set(keys) - set(cfg):
        missing, extra = set(keys) - set(cfg), set(cfg) - set(keys) - optional_keys
        details = (["missing " + ", ".join(sorted(missing))] if missing else [])
        details += (["unknown " + ", ".join(sorted(extra))] if extra else [])
        raise ValueError("config: " + "; ".join(details))
    if cfg["schema_name"] != SCHEMA:
        raise ValueError(f"schema_name must be {SCHEMA!r}")
    if not isinstance(cfg["name"], str) or not cfg["name"]:
        raise ValueError("name must be a non-empty string")
    for field, low, high in (("subjects", 1, 9), ("seeds", 0, 2**31 - 1)):
        values = cfg[field]
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or v < low or v > high for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"{field} must be unique integers in {low}..{high}")
    protocol = cfg["protocol"]
    if protocol not in PROTOCOLS:
        raise ValueError("protocol must be 'within_session' or 'cross_session'")
    split = _object(cfg["split"], "split", ("strategy", "train", "validation", "test"))
    expected_split = PROTOCOLS[protocol]
    if split != {"strategy": "stratified", **{k: expected_split[k] for k in ("train", "validation", "test")}}:
        if protocol == "cross_session":
            valid = split == {"strategy": "stratified", "train": 0.8, "validation": 0.2,
                              "test": 0.0}
        else:
            valid = False
        if not valid:
            raise ValueError(f"split settings do not match the {protocol} protocol")
    prep = _object(cfg["preprocessing"], "preprocessing",
                   ("sampling_rate", "trial_samples", "channels", "windows", "window_samples",
                    "filter_low_hz", "filter_high_hz", "filter_scope", "normalization"))
    for field in ("sampling_rate", "trial_samples", "channels", "windows", "window_samples"):
        _integer(prep[field], f"preprocessing.{field}")
    if prep["channels"] != 22 or prep["windows"] != 4:
        raise ValueError("preprocessing requires 22 channels and four windows")
    if prep["trial_samples"] != prep["windows"] * prep["window_samples"]:
        raise ValueError("trial_samples must equal windows * window_samples")
    _number(prep["filter_low_hz"], "preprocessing.filter_low_hz", 0, prep["sampling_rate"] / 2,
            low_inclusive=False)
    _number(prep["filter_high_hz"], "preprocessing.filter_high_hz", 0,
            prep["sampling_rate"] / 2, low_inclusive=False)
    if prep["filter_low_hz"] >= prep["filter_high_hz"]:
        raise ValueError("filter_low_hz must be below filter_high_hz")
    if prep["filter_scope"] != "window" or prep["normalization"] != "train_channel":
        raise ValueError("filter_scope must be 'window' and normalization must be 'train_channel'")
    model = _object(cfg["model"], "model", ("classes", "temporal_kernel", "f1", "depth_multiplier",
        "f2", "spatial_kernel", "pooling", "separable_kernel", "dropout"))
    for field in ("classes", "temporal_kernel", "f1", "depth_multiplier", "f2", "separable_kernel"):
        _integer(model[field], f"model.{field}")
    if (model["classes"] != 4 or model["spatial_kernel"] != [prep["channels"], 1]):
        raise ValueError("model requires four classes and spatial_kernel [channels, 1]")
    if model["f2"] != model["f1"] * model["depth_multiplier"]:
        raise ValueError("model.f2 must equal model.f1 * model.depth_multiplier")
    if not isinstance(model["pooling"], list) or len(model["pooling"]) != 2:
        raise ValueError("model.pooling must contain exactly two positive integers")
    for i, val in enumerate(model["pooling"]):
        _integer(val, f"model.pooling[{i}]")
    if model["pooling"][0] * model["pooling"][1] > prep["window_samples"]:
        raise ValueError("model pooling dimensions exceed window_samples")
    _number(model["dropout"], "model.dropout", 0, 1)
    if model["dropout"] >= 1:
        raise ValueError("model.dropout must be less than 1")
    training = _object(cfg["training"], "training", ("epochs", "batch_size", "learning_rate",
        "weight_decay", "warmup_epochs", "gradient_clip", "minimum_epochs", "patience"))
    for key in ("epochs", "batch_size", "minimum_epochs"):
        _integer(training[key], f"training.{key}")
    _integer(training["warmup_epochs"], "training.warmup_epochs", 0)
    _integer(training["patience"], "training.patience", 0)
    if training["minimum_epochs"] > training["epochs"] or training["warmup_epochs"] >= training["epochs"]:
        raise ValueError("minimum_epochs and warmup_epochs must fit within training.epochs")
    _number(training["learning_rate"], "training.learning_rate", 0, low_inclusive=False)
    _number(training["weight_decay"], "training.weight_decay", 0)
    _number(training["gradient_clip"], "training.gradient_clip", 0, low_inclusive=False)
    selection = _object(cfg["selection"], "selection", ("policy", "tie_breaker"))
    if selection != {"policy": "full", "tie_breaker": "validation_log_loss"}:
        if selection != {"policy": "robust", "tie_breaker": "mean_validation_log_loss"}:
            raise ValueError("selection must pair policy 'full' or 'robust' with its declared log-loss tie-breaker")
    if "head_ablation" not in cfg:
        if cfg["arms"] != [{"model": m, "regime": r} for m, r, _, _ in _ARM_DEFINITIONS]:
            raise ValueError("arms must declare exactly the four supported model/regime combinations")
    else:
        ablation = _object(cfg["head_ablation"], "head_ablation", ("arms", "temporal_head"))
        if cfg["arms"] != ablation["arms"] or ablation["arms"] != [
                {"model": "masked_eegnet", "regime": "full"},
                {"model": "masked_eegnet", "regime": "mixed"}]:
            raise ValueError("head_ablation arms must be exactly masked_eegnet/full and masked_eegnet/mixed")
        head = _object(ablation["temporal_head"], "head_ablation.temporal_head",
                       ("width", "kernel", "dilation"))
        for key in ("width", "kernel", "dilation"):
            _integer(head[key], f"head_ablation.temporal_head.{key}")
        if head["kernel"] % 2 == 0:
            raise ValueError("temporal head kernel must be odd to preserve sequence length")
    if "execution_arms" in cfg:
        names = [f"{model}__{regime}" for model, regime, _, _ in _ARM_DEFINITIONS]
        selected = cfg["execution_arms"]
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                or any(name not in names for name in selected)):
            raise ValueError("execution_arms must be a non-empty unique subset of declared arm names")
    _integer(cfg["mask_repeats"], "mask_repeats")
    _integer(cfg["bootstrap_repeats"], "bootstrap_repeats")
    if cfg["external_labels_dir"] is not None:
        cfg["external_labels_dir"] = _resolve_path(cfg["external_labels_dir"], config_path.parent,
                                                   "external_labels_dir")
    elif protocol == "cross_session":
        # A later loader must receive these labels explicitly; the config remains declarable.
        cfg["external_labels_required"] = True
    cfg["config_path"] = str(config_path)
    cfg["data_dir"] = _resolve_path(cfg["data_dir"], config_path.parent, "data_dir")
    cfg["output_dir"] = _resolve_path(cfg["output_dir"], config_path.parent, "output_dir")
    cfg["protocol_description"] = PROTOCOLS[protocol]["description"]
    return cfg


def tasks(cfg) -> list[tuple[int, int]]:
    return [(subject, seed) for subject in cfg["subjects"] for seed in cfg["seeds"]]


def arms(cfg) -> list[dict]:
    selected = cfg.get("execution_arms")
    if "head_ablation" in cfg:
        result = []
        settings = cfg["head_ablation"]["temporal_head"]
        for base in cfg["head_ablation"]["arms"]:
            model, regime = base["model"], base["regime"]
            for head_type in ("flatten", "temporal_conv"):
                suffix = "head_flatten" if head_type == "flatten" else (
                    f"head_temporal_conv_w{settings['width']}_k{settings['kernel']}_d{settings['dilation']}")
                result.append({"name": f"{model}__{regime}__{suffix}",
                    "model": model, "regime": regime, "family": "neural",
                    "coverage": _NEURAL_COVERAGE[:], "mask_conditioned": True,
                    "head_type": head_type, "temporal_head": settings.copy()})
        return [arm for arm in result if selected is None or arm["name"] in selected]
    return [{"name": f"{model}__{regime}", "model": model, "regime": regime,
             "family": family, "coverage": coverage[:], "mask_conditioned": model == "masked_eegnet"}
            for model, regime, family, coverage in _ARM_DEFINITIONS
            if selected is None or f"{model}__{regime}" in selected]
