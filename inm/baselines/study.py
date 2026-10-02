"""Single-task execution and synthetic smoke orchestration for baselines-v1."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

from .protocol import arms, tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _source_hashes() -> dict:
    package_dir = Path(__file__).resolve().parent
    root = package_dir.parents[1]
    paths = sorted(package_dir.glob("*.py")) + [
        root / "inm" / "availability.py", root / "inm" / "training.py",
        root / "inm" / "data.py", root / "agfl" / "reproducibility.py",
        root / "agfl" / "optimization.py",
    ]
    # Dataset loading and splitting are part of the real-input identity too.
    paths.extend(sorted((root / "agfl" / "datasets").rglob("*.py")))
    return {str(path.relative_to(root)): _sha256(path) for path in paths if path.is_file()}


def smoke_output_path(cfg: dict) -> Path:
    """Return a synthetic sibling path tied to the exact implementation."""
    stamp = hashlib.sha256(json.dumps(_source_hashes(), sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg["output_dir"]).expanduser().resolve().with_name(
        Path(cfg["output_dir"]).name + f"-synthetic-smoke-{stamp}")


def _package_versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "torch", "scikit-learn", "mne"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _config_identity(cfg: dict) -> dict:
    # Runtime fields are resolved by protocol.load_config and form part of the
    # manifest so a config copied to a different path cannot reuse this study.
    config = {key: value for key, value in cfg.items() if key != "external_labels_required"}
    try:
        raw_hash = _sha256(Path(cfg["config_path"]))
    except (KeyError, OSError):
        raw_hash = None
    return {"config": config, "config_sha256": raw_hash, "source_sha256": _source_hashes()}


def _input_checksums(cfg: dict, subjects) -> dict:
    sessions = ("T",) if cfg["protocol"] == "within_session" else ("T", "E")
    entries = {}
    for subject in subjects:
        for session in sessions:
            path = Path(cfg["data_dir"]) / f"A{subject:02d}{session}.gdf"
            entries[str(path)] = _sha256(path) if path.is_file() else None
    labels = cfg.get("external_labels_dir")
    if labels:
        directory = Path(labels)
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            entries[str(path)] = _sha256(path)
    return entries


def _manifest(output: Path, cfg: dict, subjects, *, synthetic: bool) -> dict:
    identity = _config_identity(cfg)
    manifest_path = output / "study.json"
    expected = {"format": "agfl-baseline-study-v1", "identity": identity,
                "package_versions": _package_versions(), "synthetic": synthetic,
                "input_sha256": ({"synthetic_fixture": "generated-per-task"} if synthetic
                                 else _input_checksums(cfg, cfg["subjects"])),
                "tasks": [[int(s), int(seed)] for s, seed in tasks(cfg)],
                "arms": [arm["name"] for arm in arms(cfg)]}
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Study manifest is unreadable at {manifest_path}; use a new output directory") from error
        if current != expected:
            raise RuntimeError(f"Study configuration, source, data, or synthetic identity changed at {manifest_path}; use a new output directory")
    else:
        # A non-empty output without its provenance root may contain another
        # study's files. Never claim or overwrite it.
        if any(output.iterdir()):
            raise RuntimeError(f"Output directory is non-empty but has no study manifest: {output}")
        _atomic_json(manifest_path, expected)
    return expected


def _task_name(subject: int, seed: int) -> str:
    return f"A{subject:02d}_seed_{seed}"


def _evaluation_banks(prepared, cfg, subject: int, seed: int, repeats: int):
    import numpy as np
    from inm.availability import evaluation_scenarios, make_mask_bank, mask_bank_digest

    banks = []
    for partition in ("validation", "test"):
        indices = prepared.split_indices[partition]
        ids = [prepared.sample_ids[index] for index in indices]
        for scenario in evaluation_scenarios():
            # The legacy mask helper labels this condition full_22; the new
            # baseline result/config contract uses the canonical name "full".
            scenario_name = "full" if scenario["retained"] == 22 else scenario["name"]
            repeat_count = 1 if scenario["retained"] == 22 else repeats
            for repeat in range(repeat_count):
                mask = make_mask_bank(len(indices), cfg["preprocessing"]["windows"],
                    scenario["retained"], scenario["pattern"], seed=seed,
                    partition=partition, subject=f"A{subject:02d}", repeat=repeat,
                    sample_ids=ids)
                banks.append({"partition": partition, "scenario": scenario_name,
                    "pattern": scenario["pattern"], "retained": scenario["retained"],
                    "mask_repeat": repeat, "mask_sha256": mask_bank_digest(mask),
                    "mask": mask, "indices": indices})
    return banks


def _evaluate(model, prepared, cfg, banks, device):
    import numpy as np
    import torch
    from inm.training import metrics

    batch_size = int(cfg["training"]["batch_size"])
    shape = (cfg["preprocessing"]["channels"], cfg["preprocessing"]["windows"],
             cfg["preprocessing"]["window_samples"])
    signals = np.asarray(prepared.signals, dtype=np.float32).reshape(-1, *shape)
    rows = []
    model.eval()
    with torch.no_grad():
        for bank in banks:
            indices = bank["indices"]
            labels = np.asarray(prepared.labels[indices], dtype=np.int64)
            counts = np.bincount(labels, minlength=4)
            if np.any(counts == 0):
                raise ValueError(f"{bank['partition']} evaluation partition lacks four classes: {counts.tolist()}")
            probability_batches = []
            for start in range(0, len(indices), batch_size):
                local = np.arange(start, min(start + batch_size, len(indices)))
                trials = indices[local]
                raw = torch.from_numpy(signals[trials]).to(device)
                mask = torch.as_tensor(bank["mask"][local], dtype=torch.bool, device=device)
                logits = model(raw, mask)
                if logits.shape != (len(local), 4) or not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Invalid or nonfinite four-class evaluation logits")
                probability_batches.append(logits.softmax(-1).cpu().numpy())
            probabilities = np.concatenate(probability_batches)
            row = metrics(labels, probabilities)
            per_class = [float(item["recall"]) for item in row.pop("per_class")]
            row.update({"partition": bank["partition"], "scenario": bank["scenario"],
                "pattern": bank["pattern"], "retained": bank["retained"],
                "mask_repeat": bank["mask_repeat"], "mask_sha256": bank["mask_sha256"],
                "sample_count": int(len(indices)), "per_class_recall": per_class})
            rows.append(row)
    return rows


def _result_complete(path: Path, identity: dict, arm_name: str, banks, *, family="neural") -> dict | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Corrupt result record at {path}; use a new output directory") from error
    if result.get("identity") != identity or result.get("arm") != arm_name:
        raise RuntimeError(f"Result identity mismatch at {path}; use a new output directory")
    expected = {(b["partition"], b["scenario"], b["mask_repeat"], b["mask_sha256"]) for b in banks}
    rows = result.get("metrics", [])
    actual = {(r.get("partition"), r.get("scenario"), r.get("mask_repeat"), r.get("mask_sha256"))
              for r in rows if isinstance(r, dict)}
    required = ("accuracy", "balanced_accuracy", "f1_macro", "per_class_recall")
    if result.get("status") != "complete" or actual != expected or len(rows) != len(expected) or any(
            any(k not in row for k in required) or
            any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
                for key in ("accuracy", "balanced_accuracy", "f1_macro")) or
            not isinstance(row["per_class_recall"], list) or len(row["per_class_recall"]) != 4 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in row["per_class_recall"]) for row in rows):
        raise RuntimeError(f"Incomplete fit/result at {path}; it cannot be reused")
    if family == "classical":
        checkpoint = path.parent / "model.npz"
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing covariance model state for completed arm at {path.parent}")
        if result.get("model_sha256") != _sha256(checkpoint):
            raise RuntimeError(f"Covariance model state corruption for completed arm at {path.parent}")
    else:
        checkpoint = path.parent / "checkpoint.pt"
        history = path.parent / "history.json"
        if not checkpoint.is_file() or not history.is_file():
            raise RuntimeError(f"Missing checkpoint or history for completed arm at {path.parent}")
        if result.get("checkpoint_sha256") != _sha256(checkpoint) or result.get("history_sha256") != _sha256(history):
            raise RuntimeError(f"Checkpoint/history corruption for completed arm at {path.parent}")
    return result


def _execute_prepared(cfg, prepared, task_dir: Path, output: Path, subject: int, seed: int,
                      selected_arms, *, device: str, synthetic: bool) -> dict:
    banks = _evaluation_banks(prepared, cfg, subject, seed, cfg["mask_repeats"])
    source_hashes = _source_hashes()
    identity = {"synthetic": synthetic, "config_sha256": _config_identity(cfg)["config_sha256"],
                "source_sha256": source_hashes, "data_fingerprint": prepared.data_fingerprint,
                "split_id": prepared.metadata.get("split_id"), "subject": subject, "seed": seed,
                "protocol": cfg["protocol"]}
    task_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in selected_arms:
        family = arm["family"]
        if family not in ("neural", "classical"):
            raise ValueError(f"Unsupported arm family {family!r} for {arm['name']}")
        arm_banks = banks if family == "neural" else [bank for bank in banks if bank["scenario"] == "full"]
        arm_dir = task_dir / "ARMS" / arm["name"]
        result_path = arm_dir / "result.json"
        cached = _result_complete(result_path, identity, arm["name"], arm_banks, family=family)
        if cached is not None:
            results.append(cached)
            continue
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise RuntimeError(f"Incomplete arm artifacts at {arm_dir}; refusing to reuse or overwrite them")
        arm_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            if family == "classical":
                from .covariance import CovarianceClassifier
                from inm.training import metrics
                import numpy as np

                model = CovarianceClassifier().fit(
                    prepared.filtered_signals[prepared.split_indices["train"]],
                    prepared.labels[prepared.split_indices["train"]])
                if model.classes_.tolist() != [0, 1, 2, 3]:
                    raise ValueError("Covariance probability columns must follow class order [0,1,2,3]")
                model_path = model.save(arm_dir / "model.npz")
                metric_rows = []
                for bank in arm_banks:
                    indices = bank["indices"]
                    labels = np.asarray(prepared.labels[indices], dtype=np.int64)
                    probabilities = model.predict_proba(prepared.filtered_signals[indices])
                    row = metrics(labels, probabilities)
                    per_class = [float(item["recall"]) for item in row.pop("per_class")]
                    row.update({"partition": bank["partition"], "scenario": "full",
                        "pattern": "full", "retained": 22, "mask_repeat": 0,
                        "mask_sha256": bank["mask_sha256"], "sample_count": int(len(indices)),
                        "per_class_recall": per_class})
                    metric_rows.append(row)
                fit_seconds = time.monotonic() - started
                extra = {"model_sha256": _sha256(model_path),
                         "class_order": model.classes_.tolist(),
                         "preprocessing": "unnormalized_filtered_signals; per-trial Ledoit-Wolf covariance; symmetric log; sqrt(2)-weighted upper triangle; StandardScaler fitted on train only",
                         "classifier": {"name": "LogisticRegression", "C": 1.0,
                             "penalty": "l2", "l1_ratio": 0.0, "solver": "lbfgs", "multiclass": "multinomial-capable",
                             "max_iter": 1000, "tol": 1e-4, "random_state": 0}}
                fit_model = None
            else:
                from .eegnet import EEGNetClassifier
                from .training import fit_classifier
                from agfl.reproducibility import seed_everything

                # Initialization must use the task seed too, independently of
                # previous arms, checkpoint loading, or ambient RNG state.
                seed_everything(seed)
                model_cfg = cfg["model"]
                prep = cfg["preprocessing"]
                model = EEGNetClassifier(channels=prep["channels"], windows=prep["windows"],
                    window_samples=prep["window_samples"], num_classes=model_cfg["classes"],
                    temporal_kernel=model_cfg["temporal_kernel"], f1=model_cfg["f1"],
                    depth_multiplier=model_cfg["depth_multiplier"], f2=model_cfg["f2"],
                    pooling=model_cfg["pooling"], separable_kernel=model_cfg["separable_kernel"],
                    dropout=model_cfg["dropout"], mask_conditioned=arm["mask_conditioned"],
                    head_type=arm.get("head_type", "flatten"),
                    temporal_head_width=arm.get("temporal_head", {}).get("width", 8),
                    temporal_head_kernel=arm.get("temporal_head", {}).get("kernel", 3),
                    temporal_head_dilation=arm.get("temporal_head", {}).get("dilation", 1))
                fit = fit_classifier(model, prepared, cfg, arm, arm_dir, device=device)
                metric_rows = _evaluate(fit["model"], prepared, cfg, banks, device)
                fit_seconds = fit["elapsed_seconds"]
                fit_model = fit["model"]
                extra = {"selected_epoch": fit["selected_epoch"],
                    "epochs_trained": fit["epochs_trained"],
                    "parameter_count": sum(p.numel() for p in fit_model.parameters()),
                    "checkpoint_sha256": _sha256(Path(fit["checkpoint_path"])),
                    "history_sha256": _sha256(Path(fit["history_path"]))}
            completed = {"format": "agfl-baseline-task-result-v1", "status": "complete",
                "synthetic": synthetic, "subject": subject, "seed": seed,
                "protocol": cfg["protocol"], "arm": arm["name"], "regime": arm["regime"],
                "head_type": arm.get("head_type", "flatten"),
                "family": family, "selection_policy": cfg["selection"]["policy"] if family == "neural" else "not_applicable",
                "coverage_scope": "full_input_only" if family == "classical" else "full_and_degraded",
                "coverage": sorted({row["scenario"] for row in metric_rows}),
                "fit_seconds": fit_seconds, "task_seconds": time.monotonic() - started,
                "identity": identity, "data_fingerprint": prepared.data_fingerprint,
                "split_id": prepared.metadata.get("split_id"), "metrics": metric_rows, **extra}
            _atomic_json(result_path, completed)
            results.append(completed)
        except Exception as error:
            _atomic_json(arm_dir / "failure.json", {"status": "failed", "arm": arm["name"],
                "error_type": type(error).__name__, "error": str(error), "identity": identity})
            raise
    return {"format": "agfl-baseline-task-v1", "status": "complete", "synthetic": synthetic,
            "subject": subject, "seed": seed, "protocol": cfg["protocol"],
            "identity": identity, "arms": results}


def run_task(cfg, task_index: int, device: str = "cuda") -> dict:
    """Prepare and execute one declared real-data subject/seed task."""
    if type(task_index) is not int or not 0 <= task_index < len(tasks(cfg)):
        raise ValueError(f"task-index must be in 0..{len(tasks(cfg)) - 1}")
    selected = arms(cfg)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if any(arm["family"] == "neural" for arm in selected):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu or provide CUDA")
    subject, seed = tasks(cfg)[task_index]
    output = Path(cfg["output_dir"]).expanduser().resolve()
    _manifest(output, cfg, [subject], synthetic=False)
    task_dir = output / "artifacts" / _task_name(subject, seed)
    from .data import prepare_subject
    try:
        prepared = prepare_subject(cfg, subject, seed, task_dir)
    except Exception as error:
        _atomic_json(task_dir / "failure.json", {"status": "failed", "subject": subject,
            "seed": seed, "synthetic": False, "error_type": type(error).__name__,
            "error": str(error)})
        raise
    return _execute_prepared(cfg, prepared, task_dir, output, subject, seed, selected,
                             device=device, synthetic=False)


def run_smoke(cfg, output_dir) -> dict:
    """Run one isolated synthetic CPU fit and evaluate every neural mask condition."""
    import copy
    from .data import make_synthetic_signal_dataset, prepare_subject

    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["training"]["epochs"] = min(2, int(smoke_cfg["training"]["epochs"]))
    smoke_cfg["training"]["minimum_epochs"] = min(smoke_cfg["training"]["minimum_epochs"],
                                                    smoke_cfg["training"]["epochs"])
    smoke_cfg["training"]["warmup_epochs"] = min(smoke_cfg["training"]["warmup_epochs"],
                                                   smoke_cfg["training"]["epochs"] - 1)
    selected = [arm for arm in arms(smoke_cfg) if arm["family"] == "neural"]
    if not selected:
        raise ValueError("Smoke requires at least one neural arm; select one with execution_arms")
    if len(tasks(smoke_cfg)) == 0:
        raise ValueError("Smoke config must declare at least one subject/seed task")
    output = Path(output_dir).expanduser().resolve()
    real_output = Path(cfg["output_dir"]).expanduser().resolve()
    if output == real_output or output in real_output.parents or real_output in output.parents:
        raise ValueError("Synthetic smoke output must be separate from and non-overlapping with the real study output")
    _manifest(output, smoke_cfg, [1], synthetic=True)
    task_dir = output / "artifacts" / "synthetic_A01_seed_0"
    if (task_dir / "dataset.json").is_file():
        raise RuntimeError(f"Synthetic smoke task artifacts already exist at {task_dir}; use a fresh output directory")
    fixture = make_synthetic_signal_dataset(seed=17, samples_per_class=15)
    prepared = prepare_subject(smoke_cfg, 1, 0, task_dir, loader=lambda _: fixture)
    prepared.metadata["synthetic"] = True
    result = _execute_prepared(smoke_cfg, prepared, task_dir, output, 1, 0, selected[:1],
                               device="cpu", synthetic=True)
    _atomic_json(task_dir / "smoke.json", {"synthetic": True, "status": "complete",
                  "arm_count": len(result["arms"]), "output_schema": result["format"]})
    return result
