"""Paired, frozen-checkpoint execution and fail-closed task artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .protocol import ARM_IDS, CONDITIONS, STRATEGY_IDS, digest, file_sha256, study_identity, tasks

SCHEMA = "agfl-completion-transformer-task-v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def initialize_output(cfg: dict) -> dict:
    """Create or verify a frozen output identity; never adopt foreign files."""
    output = Path(cfg["output_dir"]).resolve()
    from .protocol import ROOT
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output directory must not equal or contain the repository root")
    for key in ("data_dir", "candidate_source_dir", "split_source_dir"):
        protected = Path(cfg[key]).expanduser().resolve()
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"output directory overlaps protected input {key}")
    identity = study_identity(cfg)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "study-identity.json"
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"cannot verify existing study identity: {exc}") from exc
        if existing != identity:
            raise ValueError("output study identity differs from current config/source/packages; use a fresh output directory")
    else:
        occupants = [path for path in output.iterdir() if path.name != marker.name]
        if occupants:
            raise FileExistsError("output contains partial/unowned artifacts without a study identity; refusing reuse")
        _atomic_json(marker, identity)
    return identity


def _artifact_dir(output: Path, subject: int, seed: int) -> Path:
    target = output / "tasks" / f"A{subject:02d}_seed_{seed}"
    resolved = target.resolve()
    if resolved == output or output not in resolved.parents:
        raise ValueError("task output path escapes the declared output directory")
    return target


def _historical_hashes(record: dict) -> dict[str, str]:
    task_path = Path(record["task_dir"]) / "task.json"
    hashes = {"task.json": file_sha256(task_path)}
    for arm, fit in record["fits"].items():
        for name in ("result.json", "checkpoint.pt", "predictions.npz", "history.json"):
            path = fit["directory"] / name
            hashes[f"{arm}/{name}"] = file_sha256(path)
    return hashes


def _verify_complete_task(path: Path, identity: dict, *, synthetic: bool | None = None) -> dict:
    task_path = path / "task.json"
    if not task_path.is_file():
        raise ValueError("existing task is partial (task.json missing); refusing overwrite or retry")
    row = json.loads(task_path.read_text(encoding="utf-8"))
    if (row.get("schema_name") != SCHEMA or row.get("status") != "complete" or
            row.get("study_id") != identity["study_id"]):
        raise ValueError("existing task is failed, partial, or belongs to another study; refusing retry")
    if synthetic is not None and row.get("synthetic") is not synthetic:
        raise ValueError("task synthetic status does not match requested operation")
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("complete task has no artifact checksum map")
    for name, expected in artifacts.items():
        rel = Path(name)
        target = (path / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or path.resolve() not in target.parents:
            raise ValueError("unsafe task artifact path")
        if not target.is_file() or file_sha256(target) != expected:
            raise ValueError(f"task artifact checksum mismatch: {name}")
    from .reporting import verify_task_artifacts
    verify_task_artifacts(path, row)
    return row


def run_task(cfg: dict, task_index: int, *, synthetic_context: dict | None = None) -> dict:
    """Run one replay/completion task, or an explicitly synthetic injected task.

    Real mode verifies the historical record before preparation, then replays
    both backbones on full input before fitting either completer or evaluation.
    """
    identity = initialize_output(cfg)
    task_rows = tasks(cfg)
    if type(task_index) is not int or not 0 <= task_index < len(task_rows):
        raise ValueError(f"task index must be in 0..{len(task_rows)-1}")
    spec = task_rows[task_index]
    output = Path(cfg["output_dir"]).resolve()
    task_dir = _artifact_dir(output, spec["subject"], spec["seed"])
    for key in ("candidate_source_dir", "split_source_dir", "data_dir"):
        protected = Path(cfg[key]).resolve()
        if task_dir.resolve() == protected or task_dir.resolve().is_relative_to(protected) or protected.is_relative_to(task_dir.resolve()):
            raise ValueError(f"task output overlaps protected historical input {key}")
    if task_dir.exists():
        prior = _verify_complete_task(task_dir, identity, synthetic=cfg["synthetic"])
        if not cfg["synthetic"]:
            from .replay import verify_historical_task
            record = verify_historical_task(cfg, spec["subject"], spec["seed"])
            if prior.get("historical_input_sha256") != _historical_hashes(record):
                raise ValueError("historical fit/task/checkpoint inputs differ from completed task; refuse resume")
        return prior
    task_dir.mkdir(parents=True, exist_ok=False)
    synthetic = bool(cfg["synthetic"])
    status = {"schema_name": SCHEMA, "status": "running", "synthetic": synthetic,
              "subject": spec["subject"], "seed": spec["seed"], "study_id": identity["study_id"],
              "config_sha256": identity["config_sha256"], "cells": []}
    _atomic_json(task_dir / "task.json", status)
    try:
        import torch
        from inm.availability import CHANNEL_IDS, make_mask_bank, mask_bank_digest
        from inm.encoder_candidates.diagnostics import _mask_model
        from inm.encoder_candidates.training import _metrics
        from .adapters import CompletionAdapter
        from .completion import (CovarianceCompleter, TuckerCompleter,
                                 completion_state_arrays)
        if synthetic:
            if synthetic_context is None:
                raise ValueError("synthetic task requires an explicit synthetic context")
            prepared = synthetic_context["prepared"]
            models = synthetic_context["models"]
            replay = {"synthetic_untrained_backbones": True}
        else:
            from .replay import prepare_task, replay_full_input, restore_backbones
            stage = task_dir / "prepared"
            prepared, historical = prepare_task(cfg, spec["subject"], spec["seed"], stage)
            models = restore_backbones(historical)
            replay = replay_full_input(historical, prepared)

        train = prepared.train
        validation = prepared.validation
        train_raw = torch.as_tensor(np.array(train.raw, dtype=np.float32, copy=True))
        val_raw = np.asarray(validation.raw, dtype=np.float32)
        labels = np.asarray(validation.labels, dtype=np.int64)
        ids = tuple(str(value) for value in validation.sample_ids)
        if val_raw.shape != (len(labels), 22, 4, 250) or len(ids) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError("validation raw, labels and unique sample IDs are misaligned")
        if not np.isfinite(val_raw).all() or not np.isfinite(train_raw.numpy()).all():
            raise ValueError("prepared train/validation arrays must be finite")
        if set(models) != set(ARM_IDS):
            raise ValueError("task must supply exactly both fixed historical backbones")

        full_masks = np.ones((len(val_raw), 22, 4), dtype=np.bool_)
        original_full = {arm: _mask_model(models[arm], val_raw, full_masks, 32, torch.device("cpu"))
                         for arm in ARM_IDS}
        for arm in ARM_IDS:
            zero_full = _mask_model(CompletionAdapter(models[arm], "zero"), val_raw,
                                    full_masks, 32, torch.device("cpu"))
            if not np.array_equal(zero_full, original_full[arm]):
                raise ValueError(f"zero completion full-input route differs from original {arm} forward")

        covariance = CovarianceCompleter().fit(train_raw).eval()
        factor_epochs = 30
        if synthetic:
            factor_epochs = int(synthetic_context.get("factor_epochs", 1))
            if not 1 <= factor_epochs <= 30:
                raise ValueError("synthetic factor_epochs must be in 1..30")
        tucker = TuckerCompleter(epochs=factor_epochs)
        tucker_history = tucker.fit(train_raw, int(spec["seed"]))
        tucker.eval()
        completers = {"zero": None, "covariance": covariance, "tucker": tucker}
        completer_arrays = completion_state_arrays(covariance, tucker)
        _atomic_npz(task_dir / "completion-state.npz", **completer_arrays)
        _atomic_json(task_dir / "completion-history.json", tucker_history)

        rows = []
        full_probabilities: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            masks = make_mask_bank(len(val_raw), 4, condition["retained"], condition["pattern"],
                seed=int(spec["seed"]), partition="validation", subject=f"A{spec['subject']:02d}",
                repeat=0, sample_ids=ids, channel_ids=CHANNEL_IDS)
            if condition["repeats"] > 1:
                mask_repeats = [make_mask_bank(len(val_raw), 4, condition["retained"], condition["pattern"],
                    seed=int(spec["seed"]), partition="validation", subject=f"A{spec['subject']:02d}",
                    repeat=repeat, sample_ids=ids, channel_ids=CHANNEL_IDS)
                    for repeat in range(condition["repeats"])]
            else:
                mask_repeats = [masks]
            for repeat, mask_array in enumerate(mask_repeats):
                if mask_array.shape != (len(val_raw), 22, 4) or not mask_array.dtype == np.bool_ or not mask_array.any(axis=1).all():
                    raise ValueError("paired mask bank has an invalid shape/type or an all-missing window")
                mask_hash = mask_bank_digest(mask_array)
                for backbone in ARM_IDS:
                    model = models[backbone]
                    model.eval().requires_grad_(False)
                    for strategy in STRATEGY_IDS:
                        adapter = CompletionAdapter(model, strategy, completers[strategy])
                        probabilities = _mask_model(adapter, val_raw, mask_array, 32, torch.device("cpu"))
                        metric = _metrics(labels, probabilities)
                        stem = f"{backbone}__{strategy}__{condition['name']}__r{repeat}"
                        name = f"cells/{stem}.npz"
                        _atomic_npz(task_dir / name, probabilities=probabilities.astype(np.float64),
                            labels=labels, sample_ids=np.asarray(ids, dtype=np.str_),
                            mask=mask_array.astype(np.bool_), mask_sha256=np.asarray(mask_hash),
                            metric_balanced_accuracy=np.asarray(metric["balanced_accuracy"]),
                            metric_log_loss=np.asarray(metric["log_loss"]),
                            subject=np.asarray(spec["subject"], dtype=np.int64), seed=np.asarray(spec["seed"], dtype=np.int64),
                            backbone=np.asarray(backbone), strategy=np.asarray(strategy),
                            condition=np.asarray(condition["name"]), repeat=np.asarray(repeat, dtype=np.int64))
                        rows.append({"backbone": backbone, "strategy": strategy,
                            "condition": condition["name"], "repeat": repeat, "mask_sha256": mask_hash,
                            "artifact": name, **{key: metric[key] for key in ("balanced_accuracy", "log_loss")}})
                        if condition["name"] == "full_22":
                            if not np.allclose(probabilities, original_full[backbone], rtol=1e-6, atol=1e-7):
                                raise ValueError(f"full-input {strategy} probabilities differ from original {backbone}")
                            full_probabilities.setdefault(backbone, probabilities)
                            if not np.allclose(probabilities, full_probabilities[backbone], rtol=1e-6, atol=1e-7):
                                raise ValueError(f"full-input strategies differ for {backbone}")
        status.update(status="complete", replay=replay, data_id=getattr(prepared, "data_id", "synthetic"),
            split_id=getattr(prepared, "split_id", "synthetic"), validation_sample_ids=list(ids),
            validation_labels=labels.tolist(), completion_fit="train_only", classifier_fits=0,
            tucker_fit_epochs=len(tucker_history), synthetic_factor_epoch_override=(factor_epochs if synthetic else None),
            cells=rows)
        if not synthetic:
            status["historical_input_sha256"] = _historical_hashes(historical)
        artifacts = {p.relative_to(task_dir).as_posix(): file_sha256(p)
                     for p in sorted(task_dir.rglob("*")) if p.is_file() and p.name != "task.json"}
        status["artifacts"] = artifacts
        _atomic_json(task_dir / "task.json", status)
        _verify_complete_task(task_dir, identity, synthetic=synthetic)
        return status
    except BaseException as exc:
        status.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
        _atomic_json(task_dir / "task.json", status)
        raise


__all__ = ["initialize_output", "run_task"]
