"""Task orchestration and provenance-checked resume for encoder candidates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import PreparedData, PartitionData, prepare_subject
from .diagnostics import evaluate_selected, fit_local_probes
from .models import build_model
from .protocol import arms, digest, file_sha256, study_identity, tasks
from .training import fit_classifier


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


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _subject_name(subject: int) -> str:
    return f"A{subject:02d}_seed_s"


def _contained(base: Path, candidate: Path) -> Path:
    base = base.resolve()
    candidate = candidate.resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Output path escapes declared output root: {candidate}")
    return candidate


def _task_root(cfg: dict, subject: int, seed: int) -> Path:
    output_root = Path(cfg["output_dir"]).expanduser().resolve()
    target = (output_root / "tasks" / f"A{subject:02d}_seed_{seed}").resolve()
    _contained(output_root, target)
    for key in ("data_dir", "split_source_dir"):
        input_root = Path(cfg[key]).expanduser().resolve()
        if target == input_root or target in input_root.parents or input_root in target.parents:
            raise ValueError(f"Task output overlaps configured {key}: {target}")
    return target


def _task_identity(cfg: dict, prepared, declared: dict) -> dict:
    provenance = prepared.provenance
    return {"schema_name": "agfl-encoder-candidates-task-v1",
        "synthetic": bool(cfg.get("synthetic", False)),
        "partitions": ["train", "validation"], "training_regime": "full",
        # One frozen study identity is shared by every task and the report gate.
        # Task-specific verified inputs remain explicit in split_id/data_id and
        # the copied recording/metadata hashes below.
        "study_id": declared["study_id"],
        "config_sha256": declared["config_sha256"],
        "source_files_sha256": declared["source_files_sha256"],
        "packages": declared["packages"], "declared_study_id": declared["study_id"],
        "subject": int(prepared.subject), "seed": int(prepared.seed),
        "split_id": str(prepared.split_id), "data_id": str(prepared.data_id),
        "recording_sha256": provenance.get("recording_sha256", {}),
        "original_metadata_sha256": provenance.get("original_metadata_sha256", {}),
        "partition_sample_ids": provenance.get("partition_sample_ids", {
            "train": list(prepared.train.sample_ids),
            "validation": list(prepared.validation.sample_ids)}),
        "normalization": prepared.normalization,
        "channel_order": provenance.get("channel_names", []),
        "fits": [],
        "checks": {key: False for key in (
            "split_isolation", "normalization_train_only", "raw_masking",
            "checkpoint_reload", "probe_reload", "mask_pairing", "identities")}}


def _compatible_task(old: dict, expected: dict) -> None:
    keys = ("schema_name", "synthetic", "partitions", "training_regime", "study_id",
            "config_sha256", "source_files_sha256", "packages", "declared_study_id",
            "subject", "seed", "split_id", "data_id", "recording_sha256",
            "original_metadata_sha256", "partition_sample_ids", "normalization", "channel_order")
    if any(old.get(key) != expected.get(key) for key in keys):
        raise ValueError("Existing task provenance differs from current config/source/packages/inputs; use a fresh output directory")


def _verify_task_complete(path: Path, identity: dict) -> dict:
    manifest_path = path / "task.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Existing task manifest is missing or corrupt at {manifest_path}") from error
    _compatible_task(manifest, identity)
    if manifest.get("status") != "complete":
        raise ValueError("Existing task is incomplete; preserve it and restart under a fresh output directory")
    expected_check_keys = {"split_isolation", "normalization_train_only", "raw_masking",
                           "checkpoint_reload", "probe_reload", "mask_pairing", "identities"}
    if set(manifest.get("checks", {})) != expected_check_keys or not all(
            manifest["checks"].values()):
        raise ValueError("Existing task verification checks are missing or false")
    fits = manifest.get("fits")
    if not isinstance(fits, list) or [row.get("arm") for row in fits] != [
            "local_control", "local_power", "spatial_eegnet", "spatial_filterbank",
            "spatial_transformer"]:
        raise ValueError("Existing task has incomplete or unordered fit coverage")
    verified = []
    for row in fits:
        arm = row.get("arm")
        if arm not in ("local_control", "local_power", "spatial_eegnet",
                       "spatial_filterbank", "spatial_transformer"):
            raise ValueError("Existing task contains an unknown arm")
        relative = Path(row.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Existing fit result path is not task-relative")
        result_path = _contained(path, path / relative)
        if not result_path.is_file() or file_sha256(result_path) != row.get("sha256"):
            raise ValueError(f"Existing fit result checksum failed: {relative}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("status", "synthetic", "partitions", "training_regime", "subject",
                    "seed", "arm", "study_id", "config_sha256", "split_id", "data_id"):
            expected_value = identity.get(key)
            if key == "status":
                expected_value = "complete"
            if key == "arm":
                expected_value = arm
            if result.get(key) != expected_value:
                raise ValueError(f"Fit identity mismatch for {arm}: {key}")
        artifact_hashes = result.get("artifact_sha256")
        required_artifacts = {"checkpoint.pt", "history.json", "predictions.npz"}
        if arm in ("local_control", "local_power"):
            required_artifacts |= {"probe.npz", "probe_shuffled.npz"}
            if result.get("probe_converged") is not True or result.get("probe_shuffled_converged") is not True:
                raise ValueError(f"Existing local fit has incomplete probe evidence: {arm}")
        if not isinstance(artifact_hashes, dict) or not required_artifacts.issubset(artifact_hashes):
            raise ValueError(f"Existing fit has incomplete artifact hash coverage: {arm}")
        for rel, expected_hash in artifact_hashes.items():
            child = Path(rel)
            if child.is_absolute() or ".." in child.parts:
                raise ValueError("Fit artifact path is not fit-relative")
            artifact = _contained(result_path.parent, result_path.parent / child)
            if not artifact.is_file() or file_sha256(artifact) != expected_hash:
                raise ValueError(f"Fit artifact checksum failed: {arm}/{rel}")
        verified.append(row)
    if len({row["arm"] for row in verified}) != 5:
        raise ValueError("Existing task repeats a fit arm")
    return manifest


def _verify_training_normalization(prepared) -> bool:
    train = np.asarray(prepared.train.raw, dtype=np.float64)
    if train.ndim != 4 or train.shape[1:] != (22, 4, 250):
        return False
    mean = train.mean(axis=(0, 2, 3))
    std = train.std(axis=(0, 2, 3), ddof=0)
    # One training trial can have minor float32 residuals after normalization.
    return (np.max(np.abs(mean)) < 2e-5 and np.max(np.abs(std - 1.0)) < 2e-5 and
            prepared.normalization.get("axes") == ["training_trials", "time"] and
            prepared.provenance.get("normalization") == prepared.normalization)


def _verify_split(prepared) -> bool:
    train = set(prepared.train.sample_ids)
    validation = set(prepared.validation.sample_ids)
    heldout = set(prepared.provenance.get("heldout_sample_ids", []))
    return (bool(train) and bool(validation) and len(train) == len(prepared.train.sample_ids)
            and len(validation) == len(prepared.validation.sample_ids)
            and not train & validation and not train & heldout and not validation & heldout
            and not hasattr(prepared, "test"))


def _verify_raw_masking(model, prepared, device="cpu") -> bool:
    """Execute the model's masking path with hidden NaNs and compare zero fill."""
    import torch
    raw = torch.as_tensor(np.asarray(prepared.validation.raw[:1]).copy(), dtype=torch.float32,
                          device=device)
    mask = torch.ones((1, 22, 4), dtype=torch.bool, device=device)
    mask[:, 0, 0] = False
    altered = raw.clone()
    altered[:, 0, 0] = float("nan")
    model.eval()
    with torch.no_grad():
        expected = model(raw, mask)
        actual = model(altered, mask)
    return bool(torch.isfinite(actual).all() and torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))


def _predictions(model, prepared, device="cpu") -> dict:
    import torch
    model.eval()
    x = torch.as_tensor(prepared.validation.raw, dtype=torch.float32, device=device)
    mask = torch.ones(x.shape[:3], dtype=torch.bool, device=device)
    with torch.no_grad():
        probabilities = model(x, mask).softmax(dim=-1).cpu().numpy()
    return {"validation_sample_ids": np.asarray(prepared.validation.sample_ids, dtype=np.str_),
            "validation_labels": np.asarray(prepared.validation.labels, dtype=np.int64),
            "validation_probabilities": probabilities}


def run_task(cfg: dict, task_index: int, device="cpu", *, loader=None,
             synthetic_fixture: bool = False) -> dict:
    """Run one fixed task, reuse only fully verified task results, and journal failures."""
    task_list = tasks(cfg)
    if type(task_index) is not int or not 0 <= task_index < len(task_list):
        raise ValueError(f"task_index must be in 0..{len(task_list) - 1}")
    subject, seed = task_list[task_index]["subject"], task_list[task_index]["seed"]
    root = _task_root(cfg, subject, seed)
    declared = study_identity(cfg)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(root.iterdir()) and not (root / "task.json").is_file():
        raise FileExistsError(f"Unrecognized or partial task artifacts at {root}; use a fresh output directory")
    try:
        with tempfile.TemporaryDirectory(prefix="agfl-candidate-data-") as staging:
            if synthetic_fixture and cfg.get("synthetic", False) and loader is None:
                prepared = _synthetic_prepared(subject=subject, seed=seed, per_class=4)
            else:
                prepared = prepare_subject(cfg, subject, seed, Path(staging), loader=loader)
            staged_provenance = Path(staging) / "data_provenance.json"
            provenance_bytes = (staged_provenance.read_bytes() if staged_provenance.is_file()
                                else (json.dumps(prepared.provenance, sort_keys=True,
                                                 indent=2, allow_nan=False) + "\n").encode())
    except Exception as error:
        manifest_path = root / "task.json"
        if not manifest_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            failed = {"schema_name": "agfl-encoder-candidates-task-v1",
                "status": "failed", "synthetic": bool(cfg.get("synthetic", False)),
                "partitions": ["train", "validation"], "training_regime": "full",
                "subject": subject, "seed": seed, "config_sha256": declared["config_sha256"],
                "source_files_sha256": declared["source_files_sha256"],
                "packages": declared["packages"], "fits": [],
                "checks": {key: False for key in (
                    "split_isolation", "normalization_train_only", "raw_masking",
                    "checkpoint_reload", "probe_reload", "mask_pairing", "identities")},
                "failure": {"stage": "prepare_subject", "type": type(error).__name__,
                    "message": str(error), "restart_guidance": (
                        "Preserve this output, set config output_dir to a new non-overlapping path, "
                        "and rerun the explicit task index there.")}}
            _atomic_json(manifest_path, failed)
        raise
    identity = _task_identity(cfg, prepared, declared)
    manifest_path = root / "task.json"
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Existing task manifest is corrupt; preserve it and use a fresh output") from error
        _compatible_task(old, identity)
        if old.get("status") == "complete":
            completed = _verify_task_complete(root, identity)
            provenance_path = root / "data_provenance.json"
            if not provenance_path.is_file() or provenance_path.read_bytes() != provenance_bytes:
                raise ValueError("Existing task data provenance differs from current inputs")
            return completed
        raise ValueError("Existing task is incomplete; preserve it and restart under a fresh output directory")

    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "data_provenance.json"
    fd, temporary = tempfile.mkstemp(prefix=provenance_path.name + ".", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(provenance_bytes)
        os.replace(temporary, provenance_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    journal = {**identity, "status": "running", "fits": [], "checks": dict(identity["checks"])}
    _atomic_json(manifest_path, journal)
    stage = "verification"
    try:
        journal["checks"]["split_isolation"] = _verify_split(prepared)
        journal["checks"]["normalization_train_only"] = _verify_training_normalization(prepared)
        if not journal["checks"]["split_isolation"] or not journal["checks"]["normalization_train_only"]:
            raise RuntimeError("Prepared data split or training normalization verification failed")
        masks_by_arm = []
        for arm_id in arms(cfg):
            stage = f"fit:{arm_id}"
            fit_directory = _contained(root, root / "ARMS" / arm_id)
            model = build_model(arm_id, seed)
            fit = fit_classifier(model, prepared, cfg, arm_id, fit_directory, device=device)
            model = fit["model"]
            if not _verify_raw_masking(model, prepared, device=device):
                raise RuntimeError(f"Raw masking/hidden NaN verification failed for {arm_id}")
            journal["checks"]["raw_masking"] = True
            journal["checks"]["checkpoint_reload"] = True  # fit_classifier reloaded and compared probabilities.
            stage = f"evaluate:{arm_id}"
            rows = evaluate_selected(model, prepared, cfg)
            masks_by_arm.append([row["mask_sha256"] for row in rows])
            result_path = fit_directory / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            expected_fit_identity = {"status": "complete", "synthetic": bool(cfg.get("synthetic", False)),
                "partitions": ["train", "validation"], "training_regime": "full",
                "subject": subject, "seed": seed, "arm": arm_id,
                "study_id": identity["study_id"], "config_sha256": identity["config_sha256"],
                "split_id": identity["split_id"], "data_id": identity["data_id"]}
            if any(result.get(key) != value for key, value in expected_fit_identity.items()):
                raise RuntimeError(f"Fit identity does not match task provenance for {arm_id}")
            journal["checks"]["identities"] = True
            if arm_id in ("local_control", "local_power"):
                stage = f"probes:{arm_id}"
                probe_results = fit_local_probes(model, prepared, cfg, fit_directory)
                result["probe_converged"] = bool(probe_results["probe"]["converged"])
                result["probe_shuffled_converged"] = bool(probe_results["probe_shuffled"]["converged"])
                result["probe_metrics"] = {
                    "ordered": {"train": probe_results["probe"]["train_metrics"],
                                 "validation": probe_results["probe"]["validation_metrics"]},
                    "shuffled": {"train": probe_results["probe_shuffled"]["train_metrics"],
                                  "validation": probe_results["probe_shuffled"]["validation_metrics"]}}
                result["artifact_sha256"].update({
                    name + ".npz": file_sha256(fit_directory / (name + ".npz"))
                    for name in ("probe", "probe_shuffled")})
                if not result["probe_converged"] or not result["probe_shuffled_converged"]:
                    raise RuntimeError(f"A local logistic probe did not converge for {arm_id}")
                journal["checks"]["probe_reload"] = True  # diagnostic fits reload and check saved probabilities.
            result["validation"] = rows
            _atomic_npz(fit_directory / "predictions.npz", **_predictions(model, prepared, device))
            result["artifact_sha256"]["predictions.npz"] = file_sha256(fit_directory / "predictions.npz")
            result["status"] = "complete"
            result["synthetic"] = bool(cfg.get("synthetic", False))
            result["partitions"] = ["train", "validation"]
            result["training_regime"] = "full"
            _atomic_json(result_path, result)
            rel = result_path.relative_to(root).as_posix()
            journal["fits"].append({"arm": arm_id, "path": rel,
                                    "sha256": file_sha256(result_path)})
            _atomic_json(manifest_path, journal)
        stage = "pairing"
        journal["checks"]["mask_pairing"] = bool(len(masks_by_arm) == 5 and
            all(mask_hashes == masks_by_arm[0] for mask_hashes in masks_by_arm[1:]))
        journal["checks"]["identities"] = all(
            row.get("arm") == arm_id for row, arm_id in zip(journal["fits"], arms(cfg)))
        if not journal["checks"]["mask_pairing"] or not journal["checks"]["identities"]:
            raise RuntimeError("Paired masks or fit identities do not match across arms")
        if not all(journal["checks"].values()):
            raise RuntimeError("One or more task verification checks were not executed successfully")
        journal["status"] = "complete"
        journal.pop("failure", None)
        _atomic_json(manifest_path, journal)
        return journal
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        journal["failure"]["restart_guidance"] = (
            "Preserve this output, set config output_dir to a new non-overlapping path, "
            "and rerun the explicit task index there.")
        _atomic_json(manifest_path, journal)
        raise


def _synthetic_prepared(subject=1, seed=0, per_class=2) -> PreparedData:
    rng = np.random.default_rng(81031 + subject * 1009 + seed)
    labels = np.repeat(np.arange(4, dtype=np.int64), per_class)
    train = rng.normal(0, 0.05, (len(labels), 22, 4, 250)).astype(np.float32)
    validation = rng.normal(0, 0.05, (len(labels), 22, 4, 250)).astype(np.float32)
    for i, label in enumerate(labels):
        train[i, 0, 0, 0] += float(label)
        validation[i, 0, 0, 0] += float(label)
    mean = train.mean(axis=(0, 2, 3), dtype=np.float64)
    std = np.maximum(train.std(axis=(0, 2, 3), dtype=np.float64), 1e-8)
    train = ((train - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)
    validation = ((validation - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)
    train_ids = tuple(f"synthetic:A{subject:02d}:s{seed}:train:{i}" for i in range(len(labels)))
    val_ids = tuple(f"synthetic:A{subject:02d}:s{seed}:validation:{i}" for i in range(len(labels)))
    normalization = {"convention": "per_channel_train_trials_and_time_population_std",
        "mean": mean.tolist(), "std": std.tolist(), "epsilon_floor": 1e-8,
        "axes": ["training_trials", "time"]}
    provenance = {"subject": subject, "seed": seed, "split_id": "synthetic-split",
        "data_id": "synthetic-data", "recording_sha256": {},
        "original_metadata_sha256": {},
        "partition_sample_ids": {"train": list(train_ids), "validation": list(val_ids)},
        "normalization": normalization, "channel_names": [f"synthetic-ch{i}" for i in range(22)],
        "heldout_sample_ids": []}
    return PreparedData(subject, seed, PartitionData(train, labels, train_ids),
        PartitionData(validation, labels.copy(), val_ids), normalization,
        "synthetic-split", "synthetic-data", provenance)


def run_smoke(cfg: dict, output_dir=None) -> dict:
    """Train, checkpoint-reload, evaluate, and probe all five arms on synthetic CPU data."""
    import copy
    import torch
    from .protocol import ROOT

    root = Path(output_dir).expanduser().resolve() if output_dir else Path(
        tempfile.mkdtemp(prefix="agfl-encoder-candidate-smoke-")).resolve()
    forbidden = [Path(cfg[key]).expanduser().resolve() for key in
                 ("data_dir", "split_source_dir", "output_dir")]
    forbidden.append(ROOT.resolve())
    if any(root == path or root in path.parents or path in root.parents for path in forbidden):
        raise ValueError("Smoke output overlaps repository, configured inputs, or real output")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Smoke output must be a fresh directory")
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["synthetic"] = True
    smoke_cfg["subjects"] = [1]
    smoke_cfg["seeds"] = [0]
    smoke_cfg["output_dir"] = str(root)
    smoke_cfg["training"].update({"maximum_epochs": 1, "minimum_epochs": 1,
                                   "patience": 0, "batch_size": 8, "warmup_epochs": 0})
    prepared = _synthetic_prepared()
    declared = study_identity(smoke_cfg)
    identity = _task_identity(smoke_cfg, prepared, declared)
    identity["smoke"] = True
    identity["status"] = "running"
    identity["fits"] = []
    identity["checks"] = dict(identity["checks"])
    path = root / "tasks/A01_seed_0"
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "task.json"
    _atomic_json(manifest_path, identity)
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        masks_by_arm = []
        for arm_id in arms(smoke_cfg):
            fit_dir = path / "ARMS" / arm_id
            model = build_model(arm_id, 0)
            fitted = fit_classifier(model, prepared, smoke_cfg, arm_id, fit_dir, device="cpu")
            model = fitted["model"]
            identity["checks"]["checkpoint_reload"] = True
            if not _verify_raw_masking(model, prepared, "cpu"):
                raise RuntimeError(f"Synthetic hidden-NaN masking failed for {arm_id}")
            identity["checks"]["raw_masking"] = True
            rows = evaluate_selected(model, prepared, smoke_cfg)
            masks_by_arm.append([row["mask_sha256"] for row in rows])
            result_path = fit_dir / "result.json"
            result = json.loads(result_path.read_text())
            expected_fit_identity = {"status": "complete", "synthetic": True,
                "partitions": ["train", "validation"], "training_regime": "full",
                "subject": 1, "seed": 0, "arm": arm_id,
                "study_id": identity["study_id"], "config_sha256": identity["config_sha256"],
                "split_id": identity["split_id"], "data_id": identity["data_id"]}
            mismatched = [key for key, value in expected_fit_identity.items()
                          if result.get(key) != value]
            if mismatched:
                raise RuntimeError(f"Synthetic fit identity mismatch for {arm_id}: {mismatched}")
            if arm_id in ("local_control", "local_power"):
                probe_results = fit_local_probes(model, prepared, smoke_cfg, fit_dir)
                result["probe_converged"] = bool(probe_results["probe"]["converged"])
                result["probe_shuffled_converged"] = bool(probe_results["probe_shuffled"]["converged"])
                if not result["probe_converged"] or not result["probe_shuffled_converged"]:
                    raise RuntimeError(f"Synthetic local probes failed convergence for {arm_id}")
                result["artifact_sha256"].update({name + ".npz": file_sha256(fit_dir / (name + ".npz"))
                    for name in ("probe", "probe_shuffled")})
                identity["checks"]["probe_reload"] = True
            result["validation"] = rows
            _atomic_npz(fit_dir / "predictions.npz", **_predictions(model, prepared, "cpu"))
            result["artifact_sha256"]["predictions.npz"] = file_sha256(fit_dir / "predictions.npz")
            result["status"] = "complete"
            result["synthetic"] = True
            _atomic_json(result_path, result)
            identity["fits"].append({"arm": arm_id,
                "path": result_path.relative_to(path).as_posix(),
                "sha256": file_sha256(result_path)})
        identity["checks"].update({"split_isolation": _verify_split(prepared),
            "normalization_train_only": _verify_training_normalization(prepared),
            "mask_pairing": all(rows == masks_by_arm[0] for rows in masks_by_arm[1:]),
            "identities": len(identity["fits"]) == 5})
        identity["status"] = "complete" if all(identity["checks"].values()) else "failed"
        _atomic_json(manifest_path, identity)
        if identity["status"] != "complete":
            raise RuntimeError("Synthetic smoke task did not pass every execution check")
        from .reporting import summarize
        report = summarize(smoke_cfg)
        identity["report"] = report
        if report["status"] != "partial" or report["complete"]:
            raise RuntimeError("Synthetic smoke must be rejected by the real evidence gate")
        _atomic_json(manifest_path, identity)
        return identity
    except Exception as error:
        identity["status"] = "failed"
        identity["failure"] = {"type": type(error).__name__, "message": str(error)}
        _atomic_json(manifest_path, identity)
        raise
    finally:
        torch.set_num_threads(torch_threads)
