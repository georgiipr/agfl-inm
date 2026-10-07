"""Frozen protocol constants and canonical study identity helpers."""
from __future__ import annotations
import hashlib, importlib.metadata, json, platform, sys, os
from pathlib import Path
import tensorflow as tf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/published-covariance.json"
ARMS = ("zero", "covariance_fixed", "covariance_learned")
CONDITIONS = {"full_22": 1, "static_16": 5, "static_6": 5}
TASKS = [(f"A{i:02d}", seed) for i in range(1, 10) for seed in range(3)]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def runtime(device: str) -> dict:
    packages = {d.metadata.get("Name", "unknown"): d.version for d in importlib.metadata.distributions()}
    packages = dict(sorted(packages.items(), key=lambda kv: kv[0].lower()))
    gpus=tf.config.list_physical_devices("GPU")
    growth=[]
    for gpu in gpus:
        try: growth.append(bool(tf.config.experimental.get_memory_growth(gpu)))
        except RuntimeError: growth.append(None)
    try: tf_build = tf.sysconfig.get_build_info()
    except Exception: tf_build = {}
    gpu_details=[]
    for gpu in gpus:
        try: gpu_details.append(tf.config.experimental.get_device_details(gpu))
        except Exception as exc: gpu_details.append({"device":gpu.name,"details_error":type(exc).__name__})
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "python_flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site, "optimize": sys.flags.optimize},
            "platform": platform.platform(), "machine": platform.machine(), "packages": packages,
            "tensorflow_build": tf_build, "tensorflow_devices": [d.name for d in tf.config.list_physical_devices()],
            "tensorflow_logical_devices": [d.name for d in tf.config.list_logical_devices()],
            "gpu_memory_growth": growth, "gpu_details":gpu_details, "requested_device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runtime_flags": {k:os.environ.get(k) for k in ("TF_CPP_MIN_LOG_LEVEL","TF_DETERMINISTIC_OPS","TF_ENABLE_ONEDNN_OPTS","TF_CUDNN_DETERMINISTIC","TF_XLA_FLAGS","XLA_FLAGS","CUDA_VISIBLE_DEVICES")}}

def source_manifest() -> dict:
    names = sorted(p.name for p in (ROOT / "published_covariance").glob("*.py") if p.is_file())
    required={"__init__.py","__main__.py","protocol.py","study.py","audit.py","reporting.py","cli.py","native_reference.py","data.py","masks.py","completion.py","training.py","classifier.py","atcnet_native.py"}
    missing=sorted(required-set(names))
    if missing: raise RuntimeError("required source module missing: "+", ".join(missing))
    result={f"published_covariance/{name}": sha256(ROOT / "published_covariance" / name) for name in names}
    for name in ("scripts/run_published_covariance.sh", "plans/published-checkpoint-covariance/README.md", "plans/published-checkpoint-covariance/CONTRACT.md", "plans/published-checkpoint-covariance/ACCEPTANCE.md", "plans/published-checkpoint-covariance/sessions/05-execution.md", "plans/published-checkpoint-covariance/sessions/05-R1-execution-repair.md", "docs/published-covariance/checkpoint-audit.md", "docs/published-covariance/checkpoint-audit.json", "docs/published-covariance/training.md", "docs/published-covariance/data.md"):
        p=ROOT/name
        if not p.is_file(): raise RuntimeError(f"required identity input missing: {name}")
        result[name]=sha256(p)
    return result

def identity(device: str, synthetic: bool) -> dict:
    config = json.loads(CONFIG.read_text())
    value = {"study_id": config["study_id"], "configuration": config, "configuration_sha256": sha256(CONFIG), "source": source_manifest(), "runtime": runtime(device), "synthetic_fixture": bool(synthetic)}
    if not synthetic:
        asset_root = ROOT / config["real_assets"]
        origin = asset_root / "origin.json"
        value["origin_manifest_sha256"] = sha256(origin)
        value["recording_origins"] = {p.name: sha256(p) for p in sorted((asset_root / "recordings").glob("*-origin.json"))}
        origin_data = json.loads(origin.read_text())
        required={"models.py", "attention_models.py", "preprocess.py", "LICENSE"} | {f"results/saved models/run-1/subject-{i}.h5" for i in range(1,10)}
        value["published_source_and_checkpoint_sha256"] = {k: origin_data["files"][k] for k in sorted(required) if k in origin_data.get("files", {})}
        for relative,expected in value["published_source_and_checkpoint_sha256"].items():
            actual=sha256(asset_root/"EEG-ATCNet"/relative)
            if actual!=expected: raise RuntimeError(f"pinned external source/checkpoint bytes differ: {relative}")
        if set(value["published_source_and_checkpoint_sha256"]) != required: raise RuntimeError("pinned official source/checkpoint manifest must contain the exact required source files and nine H5 files")
        record_origins={}
        for p in sorted((asset_root / "recordings").glob("A0[1-9][TE]-origin.json")):
            record_origins[p.name.removesuffix("-origin.json")]=json.loads(p.read_text())["sha256"]
        if set(record_origins)!={f"A{i:02d}{s}" for i in range(1,10) for s in ("T","E")}: raise RuntimeError("the nine-subject T/E recording manifest set is incomplete")
        actual_records={}
        for key,expected in record_origins.items():
            path=asset_root/"recordings"/f"{key}.mat"
            if sha256(path)!=expected: raise RuntimeError(f"pinned native recording bytes differ: {key}")
            actual_records[key]=expected
        value["recording_bytes_sha256"] = actual_records
        alignment=ROOT/".session-runs/published-checkpoint-covariance/02/native-recording-alignment.json"
        value["native_alignment_receipt_sha256"] = sha256(alignment)
    # TensorFlow device details may contain tuples and NumPy scalars. Persist a
    # JSON-native canonical form so identity comparisons survive reload.
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
