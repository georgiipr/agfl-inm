"""Artifact verification and hierarchical paired validation summaries."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .protocol import ARM_IDS, CONDITIONS, STRATEGY_IDS, file_sha256, study_identity, tasks
from .study import SCHEMA, _verify_complete_task


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def verify_task_artifacts(task_dir: Path, task: dict) -> None:
    """Verify all six paired cell grids and recompute metrics from numeric arrays."""
    from .completion import load_completion_state
    from inm.availability import mask_bank_digest
    from inm.availability import CHANNEL_IDS, make_mask_bank
    from inm.encoder_candidates.training import _metrics

    cells = task.get("cells")
    expected_keys = {(arm, strategy, condition["name"], repeat)
        for arm in ARM_IDS for strategy in STRATEGY_IDS for condition in CONDITIONS
        for repeat in range(condition["repeats"])}
    actual_keys = {(row.get("backbone"), row.get("strategy"), row.get("condition"), row.get("repeat"))
                   for row in cells or []}
    if actual_keys != expected_keys or len(cells or []) != len(expected_keys):
        raise ValueError("task cell grid is incomplete, duplicated, or has unknown cells")
    ids = task.get("validation_sample_ids")
    labels = np.asarray(task.get("validation_labels"), dtype=np.int64)
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or len(ids) != len(labels):
        raise ValueError("task validation IDs and labels are malformed")
    hashes = task.get("artifacts")
    expected_artifacts = {row.get("artifact") for row in cells or []} | {"completion-state.npz", "completion-history.json"}
    if not isinstance(hashes, dict) or not expected_artifacts.issubset(hashes):
        raise ValueError("task artifact checksum map omits cell or completion state/history")
    actual_files = {path.relative_to(task_dir).as_posix() for path in task_dir.rglob("*") if path.is_file() and path.name != "task.json"}
    if actual_files != set(hashes):
        raise ValueError("task has unhashed or missing numeric artifacts")
    history_path = task_dir / "completion-history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completion history is unreadable: {exc}") from exc
    expected_epochs = task.get("tucker_fit_epochs")
    if type(expected_epochs) is not int or expected_epochs < 1:
        raise ValueError("task omits a valid Tucker fit epoch count")
    if task.get("synthetic") is True:
        if not 1 <= expected_epochs <= 30 or task.get("synthetic_factor_epoch_override") != expected_epochs:
            raise ValueError("synthetic Tucker epoch override is missing or out of range")
    elif task.get("synthetic") is False:
        if expected_epochs != 30 or task.get("synthetic_factor_epoch_override") is not None:
            raise ValueError("real Tucker fit must use the fixed 30 epochs")
    else:
        raise ValueError("task synthetic status is malformed")
    fixed_settings = {"channels": 22, "ridge_scale": 0.001, "windows": 4, "samples": 250,
        "rank_channels": 4, "rank_features": 16, "ridge": 0.001,
        "epochs": expected_epochs, "learning_rate": 0.01, "batch_size": 64,
        "seed_offset": 810001}
    load_completion_state(task_dir / "completion-state.npz", history,
        expected_epochs=expected_epochs, expected_settings=fixed_settings)
    seen_masks: dict[tuple, str] = {}
    full_probabilities: dict[str, np.ndarray] = {}
    for row in cells:
        rel = Path(row["artifact"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("cell artifact path must remain inside task")
        path = (task_dir / rel).resolve()
        if task_dir.resolve() not in path.parents or not path.is_file():
            raise ValueError("cell artifact is missing or escapes task directory")
        with np.load(path, allow_pickle=False) as data:
            required = {"probabilities", "labels", "sample_ids", "mask", "mask_sha256",
                "metric_balanced_accuracy", "metric_log_loss", "subject", "seed", "backbone",
                "strategy", "condition", "repeat"}
            if set(data.files) != required:
                raise ValueError(f"cell artifact arrays differ from schema: {path.name}")
            probs = data["probabilities"]
            got_labels = data["labels"]
            got_ids = data["sample_ids"].astype(str).tolist()
            mask = data["mask"]
            metadata = (str(data["backbone"].item()), str(data["strategy"].item()),
                        str(data["condition"].item()), int(data["repeat"].item()))
            if metadata != (row["backbone"], row["strategy"], row["condition"], row["repeat"]):
                raise ValueError("cell artifact identity differs from task row")
            if got_ids != ids or not np.array_equal(got_labels, labels):
                raise ValueError("cell validation labels or ordered IDs differ across task")
            if mask.shape != (len(ids), 22, 4) or mask.dtype != np.bool_ or not mask.any(axis=1).all():
                raise ValueError("cell mask has invalid shape/type or an all-missing window")
            mask_hash = mask_bank_digest(mask)
            if str(data["mask_sha256"].item()) != mask_hash or row.get("mask_sha256") != mask_hash:
                raise ValueError("cell mask checksum mismatch")
            pairing_key = (metadata[2], metadata[3])
            prior = seen_masks.setdefault(pairing_key, mask_hash)
            if prior != mask_hash:
                raise ValueError("mask bank differs across paired cells")
            condition = next(item for item in CONDITIONS if item["name"] == metadata[2])
            expected_mask = make_mask_bank(len(ids), 4, condition["retained"], condition["pattern"],
                seed=int(task["seed"]), partition="validation", subject=f"A{int(task['subject']):02d}",
                repeat=metadata[3], sample_ids=ids, channel_ids=CHANNEL_IDS)
            if not np.array_equal(mask, expected_mask):
                raise ValueError("cell mask differs from the declared deterministic condition")
            if probs.shape != (len(labels), 4) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
                raise ValueError("cell probability array is malformed")
            if not np.allclose(probs.sum(axis=1), 1, rtol=1e-7, atol=1e-7):
                raise ValueError("cell probabilities do not sum to one")
            if (int(data["subject"].item()), int(data["seed"].item())) != (task["subject"], task["seed"]):
                raise ValueError("cell artifact subject/seed differs from task")
            if metadata[2] == "full_22":
                prior_full = full_probabilities.setdefault(metadata[0], probs.copy())
                if not np.allclose(probs, prior_full, rtol=1e-6, atol=1e-7):
                    raise ValueError(f"full-input strategies differ for {metadata[0]}")
            metric = _metrics(labels, probs)
            if not _close(float(data["metric_balanced_accuracy"].item()), metric["balanced_accuracy"]):
                raise ValueError("saved balanced accuracy cannot be reproduced from probabilities")
            if not _close(float(data["metric_log_loss"].item()), metric["log_loss"]):
                raise ValueError("saved log loss cannot be reproduced from probabilities")
            if not _close(float(row["balanced_accuracy"]), metric["balanced_accuracy"]) or not _close(float(row["log_loss"]), metric["log_loss"]):
                raise ValueError("task row metrics differ from numeric artifact")


def _bootstrap(values: list[float], repeats: int, seed: int) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("participant bootstrap needs finite paired differences")
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(repeats, len(x)))].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def _load_task(path: Path, identity: dict, *, require_real: bool) -> dict:
    task = json.loads((path / "task.json").read_text(encoding="utf-8"))
    if require_real and task.get("synthetic") is not False:
        raise ValueError("synthetic task is excluded from real reports")
    return _verify_complete_task(path, identity, synthetic=task.get("synthetic"))


def summarize(cfg: dict) -> dict:
    """Create partial evidence from valid real tasks; synthetic tasks never enter it."""
    output = Path(cfg["output_dir"]).resolve()
    report_dir = output / "report"
    identity = study_identity(cfg)
    marker = output / "study-identity.json"
    if not marker.is_file() or json.loads(marker.read_text()) != identity:
        raise ValueError("output identity is absent or differs from current source/config; cannot summarize")
    valid, issues, synthetic_excluded = [], [], []
    for spec in tasks(cfg):
        path = output / "tasks" / f"A{spec['subject']:02d}_seed_{spec['seed']}"
        try:
            if not cfg["synthetic"]:
                from .replay import verify_historical_task
                from .study import _historical_hashes
                historical = verify_historical_task(cfg, spec["subject"], spec["seed"])
                candidate_task = json.loads((path / "task.json").read_text(encoding="utf-8"))
                if candidate_task.get("historical_input_sha256") != _historical_hashes(historical):
                    raise ValueError("historical task/fit/checkpoint hashes differ from completed replay")
            row = _load_task(path, identity, require_real=True)
            if (row.get("subject"), row.get("seed")) != (spec["subject"], spec["seed"]):
                raise ValueError("task subject/seed differs from plan")
            valid.append(row)
        except Exception as exc:
            if path.is_dir() and (path / "task.json").is_file():
                try:
                    candidate = json.loads((path / "task.json").read_text())
                    if candidate.get("synthetic") is True:
                        synthetic_excluded.append({"subject": spec["subject"], "seed": spec["seed"], "reason": "synthetic evidence excluded"})
                        continue
                except Exception:
                    pass
            issues.append({"subject": spec["subject"], "seed": spec["seed"], "reason": str(exc)})
    # Make cell rows convenient for tables while retaining every repeat and condition.
    cell_rows = []
    for task in valid:
        for row in task["cells"]:
            cell_rows.append({"subject": task["subject"], "seed": task["seed"], **row})
    complete = (not cfg["synthetic"] and not issues and len(valid) == 27 and
                list(cfg["subjects"]) == list(range(1, 10)) and list(cfg["seeds"]) == [0, 1, 2])
    conditions = [item["name"] for item in CONDITIONS]
    participant_conditions, participant_contrasts, cohort_conditions, contrasts = [], [], [], []
    metrics_by = {(r["subject"], r["seed"], r["backbone"], r["strategy"], r["condition"], r["repeat"]): r
                  for r in cell_rows}
    for subject in cfg["subjects"]:
        for seed in cfg["seeds"]:
            for backbone in ARM_IDS:
                for strategy in STRATEGY_IDS:
                    for condition in conditions:
                        rows = [r for r in cell_rows if (r["subject"], r["seed"], r["backbone"], r["strategy"], r["condition"]) ==
                                (subject, seed, backbone, strategy, condition)]
                        if not rows:
                            continue
                        for metric in ("balanced_accuracy", "log_loss"):
                            participant_conditions.append({"subject": subject, "seed": seed, "backbone": backbone,
                                "strategy": strategy, "condition": condition, "metric": metric,
                                "value": float(np.mean([r[metric] for r in rows])), "repeats": len(rows)})
    # Equal repeat weighting within condition, four degraded conditions per task,
    # then seeds within participant. Contrasts are paired at every level.
    contrast_defs = [
        ("primary_transformer_tucker_minus_zero", "spatial_transformer", "tucker", "zero"),
        ("secondary_eegnet_tucker_minus_zero", "spatial_eegnet", "tucker", "zero"),
        ("secondary_transformer_tucker_minus_covariance", "spatial_transformer", "tucker", "covariance"),
    ]
    degraded = conditions[1:]
    for name, backbone, candidate, control in contrast_defs:
        participant_values = {}
        for subject in cfg["subjects"]:
            seed_values = []
            for seed in cfg["seeds"]:
                diffs = []
                for condition in degraded:
                    c = [r["balanced_accuracy"] for r in cell_rows if (r["subject"], r["seed"], r["backbone"], r["strategy"], r["condition"]) == (subject, seed, backbone, candidate, condition)]
                    z = [r["balanced_accuracy"] for r in cell_rows if (r["subject"], r["seed"], r["backbone"], r["strategy"], r["condition"]) == (subject, seed, backbone, control, condition)]
                    if c and z:
                        diffs.append(float(np.mean(c) - np.mean(z)))
                if len(diffs) == len(degraded):
                    seed_values.append(float(np.mean(diffs)))
            if seed_values:
                participant_values[subject] = float(np.mean(seed_values))
                participant_contrasts.append({"subject": subject, "contrast": name, "gain": participant_values[subject],
                    "seeds": len(seed_values), "metric": "balanced_accuracy"})
        contrasts.append({"contrast": name, "metric": "balanced_accuracy", "participants": len(participant_values),
            "mean_gain": float(np.mean(list(participant_values.values()))) if participant_values else None,
            "positive_participants": sum(value > 0 for value in participant_values.values()),
            "participant_values": participant_values,
            "ci_95": _bootstrap(list(participant_values.values()), 2000, 20261005) if participant_values else None})
    # Difference-in-differences uses each participant's paired gain from the two backbones.
    interactions = {}
    for subject in cfg["subjects"]:
        per_seed = []
        for seed in cfg["seeds"]:
            benefits = []
            for backbone in ARM_IDS:
                condition_diffs = []
                for condition in degraded:
                    t = [r["balanced_accuracy"] for r in cell_rows if (r["subject"], r["seed"], r["backbone"], r["strategy"], r["condition"]) == (subject, seed, backbone, "tucker", condition)]
                    z = [r["balanced_accuracy"] for r in cell_rows if (r["subject"], r["seed"], r["backbone"], r["strategy"], r["condition"]) == (subject, seed, backbone, "zero", condition)]
                    if t and z:
                        condition_diffs.append(float(np.mean(t) - np.mean(z)))
                if len(condition_diffs) == len(degraded):
                    benefits.append(float(np.mean(condition_diffs)))
            if len(benefits) == 2:
                per_seed.append(benefits[1] - benefits[0])
        if per_seed:
            interactions[subject] = float(np.mean(per_seed))
            participant_contrasts.append({"subject": subject, "contrast": "interaction_transformer_minus_eegnet", "gain": interactions[subject], "seeds": len(per_seed), "metric": "balanced_accuracy"})
    contrasts.append({"contrast": "interaction_transformer_minus_eegnet", "metric": "balanced_accuracy",
        "participants": len(interactions), "mean_gain": float(np.mean(list(interactions.values()))) if interactions else None,
        "positive_participants": sum(value > 0 for value in interactions.values()), "participant_values": interactions,
        "ci_95": _bootstrap(list(interactions.values()), 2000, 20261005) if interactions else None,
        "interpretation": "difference-in-differences interaction; not proof of synergy"})

    for backbone in ARM_IDS:
        for strategy in STRATEGY_IDS:
            for condition in conditions:
                for metric in ("balanced_accuracy", "log_loss"):
                    participants = []
                    for subject in cfg["subjects"]:
                        seed_vals = []
                        for seed in cfg["seeds"]:
                            vals = [r["value"] for r in participant_conditions if r["subject"] == subject and r["seed"] == seed and r["backbone"] == backbone and r["strategy"] == strategy and r["condition"] == condition and r["metric"] == metric]
                            if vals:
                                seed_vals.append(vals[0])
                        if seed_vals:
                            participants.append(float(np.mean(seed_vals)))
                    cohort_conditions.append({"backbone": backbone, "strategy": strategy, "condition": condition,
                        "metric": metric, "value": float(np.mean(participants)) if participants else None,
                        "participants": len(participants)})
    if not complete:
        for row in contrasts:
            row["mean_gain"] = None
            row["positive_participants"] = None
            row["ci_95"] = None
            row["participant_values"] = {}
        for row in cohort_conditions:
            row["value"] = None
    primary = next(row for row in contrasts if row["contrast"] == "primary_transformer_tucker_minus_zero")
    promising = bool(complete and primary["mean_gain"] is not None and primary["mean_gain"] >= .02 and primary["positive_participants"] >= 6)
    status = "complete" if complete else "partial"
    evidence = {"schema_name": "agfl-completion-transformer-report-v1", "status": status,
        "synthetic": False, "study_id": identity["study_id"], "config_sha256": identity["config_sha256"],
        "subjects_declared": cfg["subjects"], "seeds_declared": cfg["seeds"], "valid_real_tasks": len(valid),
        "expected_tasks": len(tasks(cfg)), "synthetic_tasks_excluded": synthetic_excluded,
        "issues": issues, "primary_screening_rule_met": promising,
        "reused_validation_checkpoint_selection": True,
        "bootstrap": {"unit": "participant", "resamples": 2000, "seed": 20261005, "exploratory": True},
        "contrasts": contrasts}
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_dir / "evidence.json", json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _csv(report_dir / "per_repeat.csv", cell_rows, ["subject", "seed", "backbone", "strategy", "condition", "repeat", "mask_sha256", "balanced_accuracy", "log_loss", "artifact"])
    _csv(report_dir / "participant_conditions.csv", participant_conditions, ["subject", "seed", "backbone", "strategy", "condition", "metric", "value", "repeats"])
    _csv(report_dir / "cohort_conditions.csv", cohort_conditions, ["backbone", "strategy", "condition", "metric", "value", "participants"])
    _csv(report_dir / "participant_contrasts.csv", participant_contrasts, ["subject", "contrast", "gain", "seeds", "metric"])
    _csv(report_dir / "issues.csv", issues + synthetic_excluded, ["subject", "seed", "reason"])
    summary = ["# Completion and frozen-backbone replay report", "", f"Status: **{status}**.", "",
        f"Verified real tasks: {len(valid)} / {len(tasks(cfg))}.",
        "Only paired train/validation replay is included. Synthetic smoke evidence is excluded.",
        "Repeat masks are averaged within task, then seeds within participant, then participants equally.",
        "The participant bootstrap is exploratory, and checkpoints were selected on these validation participants.", ""]
    if complete:
        summary += ["## Paired contrasts", "", "| Contrast | Mean gain | 95% participant bootstrap interval | Positive participants |", "|---|---:|---:|---:|"]
        for row in contrasts:
            summary.append(f"| {row['contrast']} | {row['mean_gain']:.4f} | [{row['ci_95'][0]:.4f}, {row['ci_95'][1]:.4f}] | {row['positive_participants']} |")
        summary += ["", f"Primary exploratory screen (>=2 pp mean and >=6/9 positive participants): **{promising}**.",
                    "The interaction is a difference-in-differences and does not establish synergy."]
    else:
        summary += ["", "Incomplete coverage; cohort-level inference is withheld. Per-repeat valid task outcomes and missing-task reasons are retained in the CSV files."]
    _atomic_text(report_dir / "summary.md", "\n".join(summary) + "\n")
    return {"complete": complete, "status": status, "valid_tasks": len(valid), "issues": issues,
            "report_dir": str(report_dir), "primary_screening_rule_met": promising}


__all__ = ["summarize", "verify_task_artifacts"]
