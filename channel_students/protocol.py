"""Explicit experiment configuration; no covariance-study settings are inherited."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/published-covariance"
BACKEND_FILES = (
    "published_covariance/data.py", "published_covariance/masks.py",
    "published_covariance/atcnet_native.py", "published_covariance/tf_compat.py",
    "published_covariance/classifier.py", "published_covariance/assets.py",
    "published_covariance/preflight.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_configuration(experiment, path=None, subjects=None, seeds=None, retained_counts=None):
    if experiment not in ("q1", "q2"):
        raise ValueError("Select --experiment q1 or q2")
    path = Path(path) if path else ROOT / f"configs/atcnet-student-{experiment}.json"
    config = json.loads(path.read_text())
    required = {"schema_version", "experiment", "study_id", "subjects", "seeds",
                "retained_counts", "mask_repeats", "mask_namespace", "normalization",
                "selection", "training", "distillation", "bootstrap", "output", "backend_sha256"}
    if set(config) != required:
        raise ValueError("Configuration has missing or unsupported settings")
    if config["schema_version"] != 1 or config["experiment"] != experiment:
        raise ValueError("Configuration does not belong to the selected experiment")
    if subjects is not None:
        config["subjects"] = list(subjects)
    if seeds is not None:
        config["seeds"] = list(seeds)
    if retained_counts is not None:
        config["retained_counts"] = list(retained_counts)
    valid_subjects = {f"A{i:02d}" for i in range(1, 10)}
    if (not config["subjects"] or len(set(config["subjects"])) != len(config["subjects"])
            or not set(config["subjects"]) <= valid_subjects):
        raise ValueError("Declare a nonempty, unique A01..A09 cohort")
    if (not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"])
            or any(type(s) is not int or s < 0 for s in config["seeds"])):
        raise ValueError("Declare unique nonnegative integer seeds")
    counts = config["retained_counts"]
    if (not counts or any(type(k) is not int for k in counts)
            or len(set(counts)) != len(counts) or not set(counts) <= {16, 6}
            or config["mask_repeats"] != 5):
        raise ValueError("Choose separate 16/6-channel students and five repeats")
    if config["mask_namespace"] != "published-checkpoint-covariance-v1":
        raise ValueError("Keep the paired covariance mask namespace")
    if config["normalization"] != "student:T-train-only; teacher:historical-all-T":
        raise ValueError("Normalization contracts must be explicit")
    if config["selection"] != "maximum validation BA; lower log loss; earlier epoch":
        raise ValueError("Checkpoint selection must use reduced-channel T validation")
    training = config["training"]
    if set(training) != {"optimizer", "learning_rate", "betas", "epsilon", "batch_size",
                         "gradient_clip_global_norm", "max_epochs", "min_epochs",
                         "patience", "native_regularizers"}:
        raise ValueError("Training configuration has missing or unsupported settings")
    if len(training["betas"]) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in training["betas"]):
        raise ValueError("Adam needs two beta values in [0,1)")
    integer_fields = ("batch_size", "max_epochs", "min_epochs", "patience")
    if any(type(training[k]) is not int or training[k] < 1 for k in integer_fields):
        raise ValueError("Training budgets must be positive integers")
    if training["min_epochs"] > training["max_epochs"]:
        raise ValueError("Minimum epochs exceeds the epoch budget")
    for field in ("learning_rate", "epsilon", "gradient_clip_global_norm"):
        if not math.isfinite(training[field]) or training[field] <= 0:
            raise ValueError("Invalid optimizer setting: " + field)
    if training["optimizer"] != "Adam" or training["native_regularizers"] is not True:
        raise ValueError("Use Adam and the native ATCNet kernel regularizers")
    kd = config["distillation"]
    if set(kd) != {"enabled", "weight", "temperature"}:
        raise ValueError("Distillation configuration has unsupported settings")
    if type(kd["enabled"]) is not bool or kd["enabled"] != (experiment == "q2"):
        raise ValueError("Q1 forbids distillation; Q2 requires it")
    if not math.isfinite(kd["temperature"]) or kd["temperature"] <= 0:
        raise ValueError("Distillation temperature must be positive")
    if not math.isfinite(kd["weight"]) or kd["weight"] < 0:
        raise ValueError("Invalid distillation weight")
    if (experiment == "q1" and kd["weight"] != 0) or (experiment == "q2" and kd["weight"] <= 0):
        raise ValueError("The experiment selector and distillation weight disagree")
    bootstrap = config["bootstrap"]
    if (set(bootstrap) != {"draws", "seed", "unit"} or bootstrap["unit"] != "participant"
            or type(bootstrap["draws"]) is not int or bootstrap["draws"] < 1
            or type(bootstrap["seed"]) is not int or bootstrap["seed"] < 0):
        raise ValueError("Declare a participant bootstrap with positive draws and a nonnegative seed")
    expected = config["backend_sha256"]
    actual = {name: sha256(ROOT / name) for name in BACKEND_FILES}
    if expected != actual:
        raise ValueError("Shared model/data contract changed; explicitly review and repin this experiment's configuration")
    return config, path


def tasks(config):
    return [(subject, seed) for subject in config["subjects"] for seed in config["seeds"]]


def initialization_seed(subject, seed, retained):
    # Deliberately exclude q1/q2: paired students have identical initial weights.
    value = json.dumps(["atcnet-channel-student-init-v1", subject, seed, retained],
                       separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little") % (2 ** 31 - 1)


def order_seed(subject, seed, retained):
    value = json.dumps(["atcnet-channel-student-order-v1", subject, seed, retained],
                       separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:16], "little")


def asset_identity():
    source = json.loads((ROOT / "docs/published-covariance/checkpoints-origin.json").read_text())
    if (source["repository"] != "https://github.com/Altaheri/EEG-ATCNet"
            or source["revision"] != "65162fb359ea46a2f62c885a9987247ab491ee7a"):
        raise ValueError("Unexpected published teacher revision")
    origin = json.loads((ASSETS / "origin.json").read_text())
    if origin != source:
        raise ValueError("Published source/checkpoint manifest changed")
    files = {}
    required = ["models.py", "attention_models.py", "preprocess.py", "LICENSE", "README.md"]
    required += [f"results/saved models/run-1/subject-{i}.h5" for i in range(1, 10)]
    for relative in required:
        digest = sha256(ASSETS / "EEG-ATCNet" / relative)
        if digest != source["files"][relative]:
            raise ValueError("Published asset differs: " + relative)
        files[relative] = digest
    records = {}
    for line in (ROOT / "docs/published-covariance/recordings-sha256.txt").read_text().splitlines():
        digest, name = line.split()
        path = ASSETS / "recordings" / name
        receipt = json.loads(path.with_name(path.stem + "-origin.json").read_text())
        if sha256(path) != digest or receipt["sha256"] != digest or receipt["bytes"] != path.stat().st_size:
            raise ValueError("Native recording differs: " + name)
        records[Path(name).stem] = digest
    return {"repository": source["repository"], "revision": source["revision"],
            "files": files, "recordings": records}


def identity(config, config_path):
    import numpy as np
    import scipy
    import sklearn
    import h5py
    import tensorflow as tf
    from published_covariance.tf_compat import compatibility_record

    paths = sorted((ROOT / "channel_students").glob("*.py"))
    source = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    source.update(config["backend_sha256"])
    runtime = {"python": platform.python_version(), "tensorflow": tf.__version__,
               "numpy": np.__version__, "scipy": scipy.__version__,
               "sklearn": sklearn.__version__, "h5py": h5py.__version__,
               "attention": compatibility_record(), "python_optimize": sys.flags.optimize,
               "flags": {k: os.environ.get(k) for k in (
                   "TF_DETERMINISTIC_OPS", "TF_CUDNN_DETERMINISTIC", "TF_ENABLE_ONEDNN_OPTS",
                   "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}}
    value = {"schema": 1, "configuration": config, "configuration_sha256": sha256(config_path),
             "source": source, "runtime": runtime, "assets": asset_identity()}
    launcher = os.environ.get("AGFL_STUDENT_LAUNCHER")
    if launcher:
        value["launcher"] = {"sha256": sha256(launcher), "content": Path(launcher).read_text()}
    return value
