"""Fail-closed validation and paired reporting for encoder candidates."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .protocol import arms, file_sha256, study_identity, tasks

_CHECKS = {"split_isolation", "normalization_train_only", "raw_masking",
           "checkpoint_reload", "probe_reload", "mask_pairing", "identities"}
_METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "log_loss")
_CONTRASTS = (("local_power", "local_control", "primary"),
              ("spatial_filterbank", "spatial_eegnet", "secondary"),
              ("spatial_transformer", "spatial_eegnet", "secondary"))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _metric_block(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} metrics are missing")
    for key in _METRICS:
        score = _finite(value.get(key), f"{name}.{key}")
        if key == "log_loss":
            if score < 0:
                raise ValueError(f"{name}.log_loss cannot be negative")
        elif not 0 <= score <= 1:
            raise ValueError(f"{name}.{key} must be in [0,1]")
    recalls = value.get("per_class_recall")
    if not isinstance(recalls, list) or len(recalls) != 4:
        raise ValueError(f"{name}.per_class_recall must contain four values")
    for index, score in enumerate(recalls):
        score = _finite(score, f"{name}.per_class_recall[{index}]")
        if not 0 <= score <= 1:
            raise ValueError(f"{name} recall outside [0,1]")
    return value


def _verify_npz(path):
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if not arrays.files:
                raise ValueError("empty NPZ")
            for key in arrays.files:
                array = arrays[key]
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    raise ValueError(f"nonfinite array {key}")
    except Exception as error:
        raise ValueError(f"invalid numeric NPZ {path.name}: {error}") from error


def _verify_fit(task_dir, row, task, config_arms):
    arm = row.get("arm")
    if arm not in config_arms:
        raise ValueError(f"unknown arm {arm!r}")
    relative = Path(row.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("fit result path must be task-relative")
    result_path = (task_dir / relative).resolve()
    if task_dir.resolve() not in result_path.parents or not result_path.is_file():
        raise ValueError("fit result is missing or escapes its task")
    if file_sha256(result_path) != row.get("sha256"):
        raise ValueError(f"fit result checksum mismatch for {arm}")
    result = _json(result_path)
    if "test" in result:
        raise ValueError(f"test outputs are forbidden in candidate fit results ({arm})")
    expected = {"status": "complete", "synthetic": False,
        "partitions": ["train", "validation"], "training_regime": "full",
        "subject": task["subject"], "seed": task["seed"], "arm": arm,
        "study_id": task["study_id"], "config_sha256": task["config_sha256"],
        "split_id": task["split_id"], "data_id": task["data_id"],
        "recording_sha256": task["recording_sha256"],
        "original_metadata_sha256": task["original_metadata_sha256"]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"fit identity mismatch for {arm}")
    epoch = result.get("selected_epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError(f"selected epoch invalid for {arm}")
    if type(result.get("parameter_count")) is not int or result["parameter_count"] < 1:
        raise ValueError(f"parameter count invalid for {arm}")
    _metric_block(result.get("clean_train"), f"{arm}.clean_train")
    _metric_block(result.get("clean_validation"), f"{arm}.clean_validation")
    validation = result.get("validation")
    expected_rows = [("full_22", 0)] + [(scenario, repeat)
        for scenario in ("random_static_16", "dynamic_random_16", "random_static_6", "dynamic_random_6")
        for repeat in range(5)]
    if not isinstance(validation, list) or [(r.get("scenario"), r.get("repeat")) for r in validation] != expected_rows:
        raise ValueError(f"validation coverage/order invalid for {arm}")
    for index, metric in enumerate(validation):
        for name in ("balanced_accuracy", "log_loss"):
            value = _finite(metric.get(name), f"{arm}.validation[{index}].{name}")
            if name == "balanced_accuracy" and not 0 <= value <= 1:
                raise ValueError(f"balanced accuracy outside [0,1] for {arm}")
            if name == "log_loss" and value < 0:
                raise ValueError(f"negative log loss for {arm}")
        if not isinstance(metric.get("mask_sha256"), str) or len(metric["mask_sha256"]) != 64:
            raise ValueError(f"mask identity invalid for {arm}")
    hashes = result.get("artifact_sha256")
    required = {"checkpoint.pt", "history.json", "predictions.npz"}
    if arm in ("local_control", "local_power"):
        required |= {"probe.npz", "probe_shuffled.npz"}
        if result.get("probe_converged") is not True or result.get("probe_shuffled_converged") is not True:
            raise ValueError(f"probe did not converge for {arm}")
        if not isinstance(result.get("probe_metrics"), dict):
            raise ValueError(f"probe metrics missing for {arm}")
        for kind in ("ordered", "shuffled"):
            for part in ("train", "validation"):
                _metric_block(result["probe_metrics"][kind].get(part), f"{arm}.{kind}.{part}")
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError(f"required artifact hashes missing for {arm}")
    for name, expected_hash in hashes.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("artifact path must be fit-relative")
        artifact = (result_path.parent / rel).resolve()
        if result_path.parent.resolve() not in artifact.parents or not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise ValueError(f"artifact checksum mismatch for {arm}/{name}")
        if artifact.suffix == ".npz":
            _verify_npz(artifact)
    history = _json(result_path.parent / "history.json")
    if not isinstance(history, list) or not history or epoch > len(history):
        raise ValueError(f"history is incomplete for {arm}")
    for index, item in enumerate(history, 1):
        if item.get("epoch") != index:
            raise ValueError(f"history epoch sequence is invalid for {arm}")
        for field in ("learning_rate", "optimization_loss", "optimization_accuracy"):
            _finite(item.get(field), f"{arm}.history[{index}].{field}")
        _metric_block(item.get("clean_train_metrics"), f"{arm}.history[{index}].train")
        _metric_block(item.get("clean_validation_metrics"), f"{arm}.history[{index}].validation")
    selected = [row for row in history if row.get("selected") is True]
    if len(selected) != 1 or selected[0].get("epoch") != epoch:
        raise ValueError(f"selected epoch does not match history for {arm}")
    _metric_block(selected[0].get("clean_train_metrics"), f"{arm}.history.train")
    _metric_block(selected[0].get("clean_validation_metrics"), f"{arm}.history.validation")
    if result["clean_validation"]["balanced_accuracy"] != selected[0]["clean_validation_metrics"]["balanced_accuracy"]:
        raise ValueError(f"selected validation metric differs from history for {arm}")
    with np.load(result_path.parent / "predictions.npz", allow_pickle=False) as predictions:
        ids = predictions["validation_sample_ids"].astype(str).tolist()
        labels = predictions["validation_labels"]
        probs = predictions["validation_probabilities"]
        if len(ids) != len(set(ids)) or len(ids) != len(labels) or probs.shape != (len(ids), 4):
            raise ValueError(f"prediction identities/shapes invalid for {arm}")
    partition_ids = task.get("partition_sample_ids", {})
    if ids != partition_ids.get("validation"):
        raise ValueError(f"prediction validation IDs differ from task provenance for {arm}")
    if arm in ("local_control", "local_power"):
        for name in ("probe.npz", "probe_shuffled.npz"):
            with np.load(result_path.parent / name, allow_pickle=False) as probe:
                train_ids = probe["train_sample_ids"].astype(str).tolist()
                validation_ids = probe["validation_sample_ids"].astype(str).tolist()
                if (train_ids != partition_ids.get("train") or validation_ids != ids or
                        "converged" not in probe.files or not bool(np.asarray(probe["converged"]).item())):
                    raise ValueError(f"probe sample IDs differ from task provenance for {arm}/{name}")
    result["_validation_ids"] = ids
    return result


def _bootstrap(values, repeats, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap needs finite participant-level paired differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _csv(path, rows, fields):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def summarize(cfg):
    """Validate all declared real tasks and write complete or partial reports.

    Failed and synthetic evidence remains visible in the partial report and can
    never be converted into a complete session-10 evidence manifest.
    """
    output = Path(cfg["output_dir"]).resolve()
    report_dir = output / "report"
    declared_arms = arms(cfg)
    current_identity = study_identity(cfg)
    valid_tasks, issues, seen = [], [], set()
    for task_spec in tasks(cfg):
        subject, seed = task_spec["subject"], task_spec["seed"]
        key = (subject, seed)
        directory = output / "tasks" / f"A{subject:02d}_seed_{seed}"
        path = directory / "task.json"
        try:
            task = _json(path)
            if "test" in task:
                raise ValueError("test outputs are forbidden in candidate task metadata")
            if key in seen:
                raise ValueError("duplicate subject/seed task")
            seen.add(key)
            if task.get("status") != "complete" or task.get("synthetic") is not False:
                raise ValueError("failed or synthetic task is not real evidence")
            required = {"schema_name": "agfl-encoder-candidates-task-v1", "partitions": ["train", "validation"],
                        "training_regime": "full", "subject": subject, "seed": seed}
            if any(task.get(name) != value for name, value in required.items()):
                raise ValueError("task schema/cohort identity mismatch")
            if not all(isinstance(task.get(name), str) and len(task[name]) == 64 for name in ("study_id", "config_sha256", "split_id", "data_id")):
                raise ValueError("task identity digest is malformed")
            if task["config_sha256"] != file_sha256(cfg["config_path"]):
                raise ValueError("task config hash differs from the declared raw config")
            source_hashes = task.get("source_files_sha256")
            if not isinstance(source_hashes, dict) or not source_hashes or any(
                    not isinstance(name, str) or not isinstance(value, str) or len(value) != 64
                    for name, value in source_hashes.items()):
                raise ValueError("task source file hash map is malformed")
            packages = task.get("packages")
            if not isinstance(packages, dict) or not packages or any(
                    not isinstance(name, str) or not isinstance(value, str) or not value
                    for name, value in packages.items()):
                raise ValueError("task package version map is malformed")
            if (task["study_id"] != current_identity["study_id"] or
                    source_hashes != current_identity["source_files_sha256"] or
                    packages != current_identity["packages"]):
                raise ValueError("task study/source/package provenance differs from current frozen identity")
            for field in ("recording_sha256", "original_metadata_sha256"):
                hashes = task.get(field)
                if not isinstance(hashes, dict) or not hashes or any(
                        not isinstance(name, str) or not isinstance(value, str) or len(value) != 64
                        for name, value in hashes.items()):
                    raise ValueError(f"task {field} provenance is missing or malformed")
            if set(task.get("checks", {})) != _CHECKS or any(value is not True for value in task["checks"].values()):
                raise ValueError("task checks are missing or false")
            fits = task.get("fits")
            if not isinstance(fits, list) or [item.get("arm") for item in fits] != declared_arms:
                raise ValueError("task has missing, duplicate, or unordered arms")
            parsed = [_verify_fit(directory, row, task, declared_arms) for row in fits]
            ids = [item.pop("_validation_ids") for item in parsed]
            if any(item != ids[0] for item in ids[1:]):
                raise ValueError("validation sample IDs differ across paired arms")
            masks = [[row["mask_sha256"] for row in item["validation"]] for item in parsed]
            if any(mask != masks[0] for mask in masks[1:]):
                raise ValueError("validation masks differ across paired arms")
            if any((item["split_id"], item["data_id"]) != (task["split_id"], task["data_id"]) for item in parsed):
                raise ValueError("fit split/data identity differs from task")
            record = {"subject": subject, "seed": seed, "path": path.relative_to(output).as_posix(),
                      "sha256": file_sha256(path), "task": task, "fits": dict(zip(declared_arms, parsed))}
            valid_tasks.append(record)
        except Exception as error:
            issues.append({"subject": subject, "seed": seed, "reason": str(error)})

    global_ids = {(row["task"]["study_id"], row["task"]["config_sha256"],
                   json.dumps(row["task"].get("source_files_sha256"), sort_keys=True),
                   json.dumps(row["task"].get("packages"), sort_keys=True)) for row in valid_tasks}
    if len(global_ids) > 1:
        issues.append({"subject": "", "seed": "", "reason": "task source/config/package identities differ"})
        valid_tasks = []
    complete = (not issues and len(valid_tasks) == len(tasks(cfg)) and
                len(valid_tasks) == 27 and list(cfg["subjects"]) == list(range(1, 10)) and list(cfg["seeds"]) == [0, 1, 2])

    # Participant first within each seed-balanced arm metric; trial counts never weight a row.
    per_subject, contrast_rows, probe_rows, robustness_rows, gap_rows = [], [], [], [], []
    for task_record in valid_tasks:
        subject, seed = task_record["subject"], task_record["seed"]
        for arm in declared_arms:
            fit = task_record["fits"][arm]
            per_subject.append({"subject": subject, "seed": seed, "arm": arm,
                "validation_ba": fit["clean_validation"]["balanced_accuracy"],
                "validation_log_loss": fit["clean_validation"]["log_loss"],
                "parameter_count": fit["parameter_count"], "selected_epoch": fit["selected_epoch"]})
            gap_rows.append({"subject": subject, "seed": seed, "arm": arm,
                "train_ba": fit["clean_train"]["balanced_accuracy"],
                "validation_ba": fit["clean_validation"]["balanced_accuracy"],
                "ba_gap": fit["clean_train"]["balanced_accuracy"] - fit["clean_validation"]["balanced_accuracy"],
                "train_log_loss": fit["clean_train"]["log_loss"],
                "validation_log_loss": fit["clean_validation"]["log_loss"]})
            if arm in ("local_control", "local_power"):
                for kind in ("ordered", "shuffled"):
                    probe = fit["probe_metrics"][kind]["validation"]
                    probe_rows.append({"subject": subject, "seed": seed, "arm": arm, "probe": kind,
                                       "balanced_accuracy": probe["balanced_accuracy"], "log_loss": probe["log_loss"]})
            for row in fit["validation"]:
                robustness_rows.append({"subject": subject, "seed": seed, "arm": arm,
                    "scenario": row["scenario"], "repeat": row["repeat"],
                    "mask_sha256": row["mask_sha256"], "balanced_accuracy": row["balanced_accuracy"],
                    "log_loss": row["log_loss"]})
    robustness_cohort = []
    if complete:
        scenarios = ("full_22", "random_static_16", "dynamic_random_16",
                     "random_static_6", "dynamic_random_6")
        for arm in declared_arms:
            for scenario in scenarios:
                participant_ba = []
                participant_loss = []
                for subject in range(1, 10):
                    seed_ba, seed_loss = [], []
                    for seed in (0, 1, 2):
                        rows = [row for row in robustness_rows if row["subject"] == subject and
                                row["seed"] == seed and row["arm"] == arm and row["scenario"] == scenario]
                        if not rows:
                            raise ValueError("complete validation task is missing a robustness scenario")
                        seed_ba.append(float(np.mean([row["balanced_accuracy"] for row in rows])))
                        seed_loss.append(float(np.mean([row["log_loss"] for row in rows])))
                    participant_ba.append(float(np.mean(seed_ba)))
                    participant_loss.append(float(np.mean(seed_loss)))
                robustness_cohort.append({"arm": arm, "scenario": scenario,
                    "balanced_accuracy": float(np.mean(participant_ba)),
                    "log_loss": float(np.mean(participant_loss)), "participants": 9})
    if complete:
        for candidate, control, scope in _CONTRASTS:
            differences_by_metric = {}
            for metric in ("balanced_accuracy", "log_loss"):
                subject_differences = []
                for subject in range(1, 10):
                    per_seed = []
                    for seed in (0, 1, 2):
                        a = next(row for row in per_subject if row["subject"] == subject and row["seed"] == seed and row["arm"] == candidate)
                        b = next(row for row in per_subject if row["subject"] == subject and row["seed"] == seed and row["arm"] == control)
                        per_seed.append(a["validation_ba"] - b["validation_ba"] if metric == "balanced_accuracy" else b["validation_log_loss"] - a["validation_log_loss"])
                    subject_differences.append(float(np.mean(per_seed)))
                    contrast_rows.append({"scope": "participant", "contrast": f"{candidate} - {control}",
                        "kind": scope, "subject": subject, "metric": metric, "gain": subject_differences[-1]})
                differences_by_metric[metric] = subject_differences
            for metric, diffs in differences_by_metric.items():
                ci_metric = _bootstrap(diffs, int(cfg["comparisons"]["bootstrap_repeats"]), int(cfg["comparisons"]["bootstrap_seed"]))
                promising = (metric == "balanced_accuracy" and
                    float(np.mean(diffs)) >= cfg["comparisons"]["screening_mean_gain"] and
                    sum(value > 0 for value in diffs) >= cfg["comparisons"]["screening_positive_participants"])
                contrast_rows.append({"scope": "cohort", "contrast": f"{candidate} - {control}", "kind": scope,
                    "subject": "", "metric": metric, "gain": float(np.mean(diffs)), "ci_low": ci_metric[0],
                    "ci_high": ci_metric[1], "positive_participants": sum(value > 0 for value in diffs),
                    "promising_for_confirmation": promising})

    report_identity = next(iter(global_ids), None)
    evidence = {"schema_name": "agfl-encoder-candidates-report-v1", "status": "complete" if complete else "partial",
        "synthetic": False, "partitions": ["train", "validation"], "training_regime": "full",
        "subjects": list(cfg["subjects"]), "seeds": list(cfg["seeds"]), "arms": declared_arms,
        "study_id": report_identity[0] if report_identity else "", "config_sha256": report_identity[1] if report_identity else "",
        "packages": json.loads(report_identity[3]) if report_identity else {},
        "source_files_sha256": json.loads(report_identity[2]) if report_identity else {},
        "tasks": [{"subject": row["subject"], "seed": row["seed"], "path": row["path"], "sha256": row["sha256"]} for row in valid_tasks] if complete else []}
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_dir / "evidence.json", json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _csv(report_dir / "validation_contrasts.csv", contrast_rows, ["scope", "contrast", "kind", "subject", "metric", "gain", "ci_low", "ci_high", "positive_participants", "promising_for_confirmation"])
    _csv(report_dir / "per_subject_scores.csv", per_subject, ["subject", "seed", "arm", "validation_ba", "validation_log_loss", "parameter_count", "selected_epoch"])
    _csv(report_dir / "training_gaps.csv", gap_rows, ["subject", "seed", "arm", "train_ba", "validation_ba", "ba_gap", "train_log_loss", "validation_log_loss"])
    _csv(report_dir / "probes.csv", probe_rows, ["subject", "seed", "arm", "probe", "balanced_accuracy", "log_loss"])
    _csv(report_dir / "robustness.csv", robustness_rows, ["subject", "seed", "arm", "scenario", "repeat", "mask_sha256", "balanced_accuracy", "log_loss"])
    _csv(report_dir / "robustness_cohort.csv", robustness_cohort,
         ["arm", "scenario", "balanced_accuracy", "log_loss", "participants"])
    _csv(report_dir / "issues.csv", issues, ["subject", "seed", "reason"])
    status = "complete" if complete else "partial"
    text = ["# Encoder candidate report", "", f"Status: **{status}**.", "",
        "Only validation outcomes are reported. Cohort contrasts are withheld unless all 27 real tasks and all five paired arms validate.",
        "Participant bootstrap intervals are exploratory (2000 resamples, seed 20261003); comparisons reuse validation for epoch selection and include multiple comparisons.",
        "The promising rule is a screening threshold for independent confirmation, not significance or proof of BCI utility.", ""]
    if complete:
        text += ["## Cohort contrasts", "", "| Contrast | Validation metric | Mean paired gain | Exploratory 95% interval | Positive participants | Promising for confirmation |", "|---|---:|---:|---:|---:|---|"]
        for row in contrast_rows:
            if row["scope"] == "cohort":
                text.append(f"| {row['contrast']} | {row['metric']} | {row['gain']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['positive_participants']} | {row['promising_for_confirmation']} |")
    else:
        text += ["Valid task records retained: " + str(len(valid_tasks)), "", "Cohort means are withheld because evidence is incomplete or invalid."]
    _atomic_text(report_dir / "summary.md", "\n".join(text) + "\n")
    return {"complete": complete, "status": status, "report_dir": str(report_dir),
            "evidence_path": str(report_dir / "evidence.json"), "issues": issues,
            "valid_tasks": len(valid_tasks)}
