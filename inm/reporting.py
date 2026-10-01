"""Atomic, completeness-aware reports from saved experiment records only.

The cluster calls ``summarize`` after each fit. Nothing is inferred from a
partially written or malformed result, and missing fits never become zeros.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .availability import DEGRADED_PATTERNS, evaluation_scenarios
from .protocol import arms, task_name, tasks


METRICS = ("balanced_accuracy", "accuracy", "f1_macro")
PARTITIONS = ("validation", "test")
IDENTITY_HASHES = ("dataset_fingerprint", "split_id", "calibration_sha256")


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _sd(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _expected_cells(cfg: dict) -> dict[tuple, dict]:
    return {
        (partition, scenario["name"], repeat): scenario
        for partition in PARTITIONS
        for scenario in evaluation_scenarios()
        for repeat in range(1 if scenario["retained"] == 22 else cfg["mask_repeats"])
    }


def _validated_result(
    value: object, subject: int, seed: int, arm: dict, study_id: str, expected: dict,
) -> dict[tuple, dict]:
    if not isinstance(value, dict):
        raise ValueError("result must be a JSON object")
    if value.get("status") != "complete" or value.get("study_id") != study_id:
        raise ValueError("result status or study identity does not match")
    if value.get("subject") != subject or value.get("seed") != seed:
        raise ValueError("result subject or seed does not match its declared task")
    identity = value.get("identity")
    if not isinstance(identity, dict) or any(
        identity.get(key) != expected_value
        for key, expected_value in (("study_id", study_id), ("subject", subject), ("seed", seed))
    ):
        raise ValueError("result identity must contain matching study_id, subject and seed")
    if any(not _is_sha256(identity.get(key)) for key in IDENTITY_HASHES):
        raise ValueError("result identity needs dataset_fingerprint, split_id and calibration_sha256 hashes")
    if not isinstance(value.get("arm"), dict) or any(
        value["arm"].get(key) != arm[key] for key in ("name", "attention", "representation", "regime")
    ):
        raise ValueError("result arm does not match its declared configuration")
    if not isinstance(value.get("selection"), dict):
        raise ValueError("result lacks checkpoint-selection metadata")
    rows = value.get("metrics")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"result must contain exactly {len(expected)} metric rows")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every metric row must be an object")
        repeat = row.get("mask_repeat")
        if type(repeat) is not int:
            raise ValueError("mask_repeat must be an integer")
        key = (row.get("partition"), row.get("scenario"), repeat)
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate partition/scenario/repeat cell")
        scenario = expected[key]
        if row.get("pattern") != scenario["pattern"] or row.get("retained") != scenario["retained"]:
            raise ValueError("scenario pattern/count metadata does not match")
        mask_hash = row.get("mask_sha256")
        if not _is_sha256(mask_hash):
            raise ValueError("missing or malformed mask SHA256")
        for metric in METRICS:
            score = row.get(metric)
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0.0 <= score <= 1.0):
                raise ValueError(f"{metric} must be finite and in [0,1]")
        nrmse = row.get("reconstruction_nrmse")
        if nrmse is not None and (
            isinstance(nrmse, bool) or not isinstance(nrmse, (int, float))
            or not math.isfinite(nrmse) or nrmse < 0.0
        ):
            raise ValueError("reconstruction_nrmse must be nonnegative and finite when supplied")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError("result is missing required metric identities")
    return indexed


def _scenario_scores(indexed: dict, partition: str, scenario: str) -> dict:
    rows = [row for (part, name, _), row in indexed.items()
            if part == partition and name == scenario]
    scores = {metric: _mean([float(row[metric]) for row in rows]) for metric in METRICS}
    nrmse = [row.get("reconstruction_nrmse") for row in rows]
    scores["reconstruction_nrmse"] = (
        _mean([float(value) for value in nrmse]) if all(value is not None for value in nrmse) else None
    )
    return scores


def _run_table(records: dict, payloads: dict) -> list[dict]:
    """Each validated fit remains inspectable before the other seeds finish."""
    rows = []
    for key, indexed in records.items():
        payload = payloads[key]
        arm = payload["arm"]
        for partition in PARTITIONS:
            for scenario in evaluation_scenarios():
                cells = [row for (part, name, _), row in sorted(indexed.items())
                         if part == partition and name == scenario["name"]]
                scores = _scenario_scores(indexed, partition, scenario["name"])
                full = _scenario_scores(indexed, partition, "full_22")
                row = {
                    "study_id": payload["study_id"], "subject": payload["subject"], "seed": payload["seed"],
                    "arm": arm["name"], "attention": arm["attention"],
                    "representation": arm["representation"], "regime": arm["regime"],
                    "partition": partition, "scenario": scenario["name"], "pattern": scenario["pattern"],
                    "retained": scenario["retained"], "mask_repeats": len(cells),
                    "best_epoch": payload["selection"].get("best_epoch"),
                    "classifier_trainable_parameters": payload.get("classifier_trainable_parameters"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    **{name: payload["identity"][name] for name in IDENTITY_HASHES},
                    "mask_sha256": ";".join(cell["mask_sha256"] for cell in cells),
                }
                for metric in METRICS:
                    row[f"{metric}_percent"] = 100.0 * scores[metric]
                    row[f"{metric}_degradation_pp"] = 100.0 * (full[metric] - scores[metric])
                row["reconstruction_nrmse"] = scores["reconstruction_nrmse"]
                rows.append(row)
    return rows


def _subject_tables(records: dict, cfg: dict, declared_arms: list[dict]) -> tuple[list, dict]:
    rows = []
    indexed = {}
    for arm in declared_arms:
        for subject in cfg["subjects"]:
            seed_records = [records[(subject, seed, arm["name"])] for seed in cfg["seeds"]
                            if (subject, seed, arm["name"]) in records]
            complete = len(seed_records) == len(cfg["seeds"])
            for partition in PARTITIONS:
                for scenario in evaluation_scenarios():
                    row = {
                        "subject": subject, "arm": arm["name"],
                        "attention": arm["attention"], "representation": arm["representation"],
                        "regime": arm["regime"], "partition": partition,
                        "scenario": scenario["name"], "pattern": scenario["pattern"],
                        "retained": scenario["retained"], "complete": complete,
                        "seeds_completed": len(seed_records), "seeds_expected": len(cfg["seeds"]),
                        "mask_repeats": 1 if scenario["retained"] == 22 else cfg["mask_repeats"],
                    }
                    scores = [_scenario_scores(record, partition, scenario["name"])
                              for record in seed_records] if complete else []
                    full = [_scenario_scores(record, partition, "full_22")
                            for record in seed_records] if complete else []
                    for metric in METRICS:
                        values = [100.0 * score[metric] for score in scores]
                        row[f"{metric}_percent"] = _mean(values) if complete else None
                        row[f"{metric}_seed_sd_pp"] = _sd(values) if complete else None
                        row[f"{metric}_degradation_pp"] = _mean([
                            100.0 * (base[metric] - score[metric]) for base, score in zip(full, scores)
                        ]) if complete else None
                    nrmse = [score["reconstruction_nrmse"] for score in scores]
                    row["reconstruction_nrmse"] = (
                        _mean(nrmse) if complete and all(value is not None for value in nrmse) else None
                    )
                    rows.append(row)
                    indexed[(subject, arm["name"], partition, scenario["name"])] = row
    return rows, indexed


def _overall_tables(by_subject: dict, cfg: dict, declared_arms: list[dict]) -> list[dict]:
    rows = []
    for arm in declared_arms:
        for partition in PARTITIONS:
            for scenario in evaluation_scenarios():
                subjects = [by_subject[(subject, arm["name"], partition, scenario["name"])]
                            for subject in cfg["subjects"]]
                complete = all(item["complete"] for item in subjects)
                row = {
                    "arm": arm["name"], "attention": arm["attention"],
                    "representation": arm["representation"], "regime": arm["regime"],
                    "partition": partition, "scenario": scenario["name"],
                    "pattern": scenario["pattern"], "retained": scenario["retained"],
                    "complete": complete,
                    "subjects_completed": sum(item["complete"] for item in subjects),
                    "subjects_expected": len(subjects),
                    "fits_completed": sum(item["seeds_completed"] for item in subjects),
                    "fits_expected": len(subjects) * len(cfg["seeds"]),
                }
                for metric in METRICS:
                    values = [item[f"{metric}_percent"] for item in subjects] if complete else []
                    row[f"{metric}_mean_percent"] = _mean(values) if complete else None
                    row[f"{metric}_subject_sd_pp"] = _sd(values) if complete else None
                    row[f"{metric}_degradation_pp"] = _mean([
                        item[f"{metric}_degradation_pp"] for item in subjects
                    ]) if complete else None
                nrmse = [item["reconstruction_nrmse"] for item in subjects]
                row["reconstruction_nrmse"] = (
                    _mean(nrmse) if complete and all(value is not None for value in nrmse) else None
                )
                rows.append(row)
    return rows


def _robustness_tables(by_subject: dict, cfg: dict, declared_arms: list[dict]) -> list[dict]:
    rows = []
    for arm in declared_arms:
        for partition in PARTITIONS:
            for pattern in DEGRADED_PATTERNS:
                names = ["full_22", *[f"{pattern}_{k}" for k in (16, 11, 6)]]
                subject_values = []
                for subject in cfg["subjects"]:
                    cells = [by_subject[(subject, arm["name"], partition, name)] for name in names]
                    if all(cell["complete"] for cell in cells):
                        subject_values.append({
                            metric: _mean([cell[f"{metric}_percent"] for cell in cells])
                            for metric in METRICS
                        })
                complete = len(subject_values) == len(cfg["subjects"])
                row = {
                    "arm": arm["name"], "attention": arm["attention"],
                    "representation": arm["representation"], "regime": arm["regime"],
                    "partition": partition, "pattern": pattern, "retained_counts": "22;16;11;6",
                    "count_weights": "0.25;0.25;0.25;0.25", "complete": complete,
                    "subjects_completed": len(subject_values), "subjects_expected": len(cfg["subjects"]),
                }
                for metric in METRICS:
                    values = [subject[metric] for subject in subject_values] if complete else []
                    row[f"{metric}_mean_percent"] = _mean(values) if complete else None
                    row[f"{metric}_subject_sd_pp"] = _sd(values) if complete else None
                rows.append(row)
    return rows


def _paired_tables(records: dict, cfg: dict, study_id: str) -> tuple[list[dict], list[dict]]:
    rows = []
    issues = []
    subjects = cfg["subjects"]
    bootstrap_seed = int.from_bytes(hashlib.sha256(
        (study_id + "subject-bootstrap").encode("utf-8")
    ).digest()[:8], "little")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(subjects), size=(cfg["bootstrap_repeats"], len(subjects)))
    specifications = [
        {"kind": "scenario", "scenario": item["name"], "pattern": item["pattern"],
         "retained": item["retained"], "names": [item["name"]]}
        for item in evaluation_scenarios()
    ] + [
        {"kind": "robustness", "scenario": f"robustness_{pattern}", "pattern": pattern,
         "retained": "22;16;11;6", "names": ["full_22", *[f"{pattern}_{k}" for k in (16, 11, 6)]]}
        for pattern in DEGRADED_PATTERNS
    ]
    for attention in cfg["attentions"]:
        for regime in cfg["regimes"]:
            base_name = f"{attention}__baseline__{regime}"
            tensor_name = f"{attention}__tensor_core__{regime}"
            paired = {}
            for subject, seed in tasks(cfg):
                base = records.get((subject, seed, base_name))
                tensor = records.get((subject, seed, tensor_name))
                if base is None or tensor is None:
                    continue
                mismatched = [key for key in base if base[key]["mask_sha256"] != tensor[key]["mask_sha256"]]
                if mismatched:
                    issues.append({
                        "subject": subject, "seed": seed, "attention": attention, "regime": regime,
                        "reason": "baseline/tensor masks differ; pair excluded from all paired tables",
                        "mismatched_cells": [list(key) for key in mismatched],
                    })
                    continue
                paired[(subject, seed)] = (base, tensor)
            for partition in PARTITIONS:
                for specification in specifications:
                    subject_differences = []
                    for subject in subjects:
                        if not all((subject, seed) in paired for seed in cfg["seeds"]):
                            continue
                        seed_differences = []
                        for seed in cfg["seeds"]:
                            base, tensor = paired[(subject, seed)]
                            # First average repeats within each scenario, then
                            # count conditions, then seeds within the subject.
                            differences = []
                            for name in specification["names"]:
                                base_score = _scenario_scores(base, partition, name)
                                tensor_score = _scenario_scores(tensor, partition, name)
                                differences.append({metric: 100.0 * (tensor_score[metric] - base_score[metric])
                                                    for metric in METRICS})
                            seed_differences.append({metric: _mean([value[metric] for value in differences])
                                                     for metric in METRICS})
                        subject_differences.append({metric: _mean([value[metric] for value in seed_differences])
                                                    for metric in METRICS})
                    complete = len(subject_differences) == len(subjects)
                    row = {
                        "attention": attention, "regime": regime, "partition": partition,
                        "baseline_arm": base_name, "tensor_arm": tensor_name,
                        "kind": specification["kind"], "scenario": specification["scenario"],
                        "pattern": specification["pattern"], "retained": specification["retained"],
                        "complete": complete, "paired_subjects_completed": len(subject_differences),
                        "subjects_expected": len(subjects), "paired_seeds_completed": len(paired),
                        "paired_seeds_expected": len(subjects) * len(cfg["seeds"]),
                        "bootstrap_repeats": cfg["bootstrap_repeats"],
                    }
                    for metric in METRICS:
                        values = [value[metric] for value in subject_differences] if complete else []
                        row[f"{metric}_gain_pp"] = _mean(values) if complete else None
                        row[f"{metric}_subject_sd_pp"] = _sd(values) if complete else None
                        interval = (None, None)
                        if complete and len(subjects) > 1:
                            samples = np.asarray(values, dtype=np.float64)[indices].mean(axis=1)
                            bounds = np.quantile(samples, [0.025, 0.975])
                            interval = (float(bounds[0]), float(bounds[1]))
                        row[f"{metric}_ci95_low_pp"], row[f"{metric}_ci95_high_pp"] = interval
                    rows.append(row)
    return rows, issues


def _format(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2f}"


def _summary_text(progress: dict, overall: list[dict], paired: list[dict]) -> str:
    lines = [
        "# AGFL-inm experiment report", "",
        f"Study/source identity: `{progress['study_id']}`.", "",
        f"**{progress['completed_fits']}/{progress['expected_fits']} fits complete.** "
        f"Status: {progress['status']}. Invalid result files: {len(progress['invalid_results'])}. "
        f"Recorded error files: {progress['failed_fit_count']}.", "",
        "Every complete fit must contain both validation and test, the full-input scenario, "
        "and all declared degraded scenarios and mask repeats. Subject scores require all "
        "configured seeds; overall means require all configured subjects. Incomplete means "
        "are blank in CSV files and shown as pending here, never treated as zero. "
        "Use accuracy_by_run.csv for completed individual fits while the other seeds are pending.", "",
        "Balanced accuracy is the primary outcome. Accuracy and macro-F1 are secondary. "
        "Full-input validation selects checkpoints for every training regime; degraded "
        "validation is evaluated only after selection. Held-out test labels do not select "
        "epochs, tensor factors or arms.", "",
        "## Full-input held-out test performance", "",
        "Scores are percentages, averaged over mask repeats, then seeds within each subject, "
        "then equally over configured subjects. SD is across subject means, not independent seeds.", "",
        "| Attention | Representation | Training | Subjects | Balanced accuracy | Subject SD | Accuracy | Macro-F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        if row["partition"] == "test" and row["scenario"] == "full_22":
            lines.append(
                f"| {row['attention']} | {row['representation']} | {row['regime']} | "
                f"{row['subjects_completed']}/{row['subjects_expected']} | "
                f"{_format(row['balanced_accuracy_mean_percent'])} | "
                f"{_format(row['balanced_accuracy_subject_sd_pp'])} | "
                f"{_format(row['accuracy_mean_percent'])} | {_format(row['f1_macro_mean_percent'])} |"
            )
    lines += [
        "", "## Paired robustness: tensor core minus baseline", "",
        "Within each loss pattern, robustness equally weights 22, 16, 11 and 6 retained "
        "channels (25% each). Each degraded condition first averages its mask repeats. "
        "Pairs must have identical subject, seed, regime, attention, scenario/repeat identities "
        "and mask hashes. Every arm within a subject/seed task must also share dataset, "
        "split and calibration hashes; conflicting tasks are excluded. Positive differences "
        "favor the tensor route.", "",
        "The 95% percentile bootstrap resamples subject means after averaging paired seed "
        "differences within each subject. These are pointwise exploratory intervals, with no "
        "correction for multiple comparisons. No interval is reported for a single subject.", "",
        "| Attention | Training | Pattern | Paired subjects | Balanced accuracy gain (pp) | 95% CI (pp) | Accuracy gain (pp) |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in paired:
        if row["partition"] == "test" and row["kind"] == "robustness":
            low, high = row["balanced_accuracy_ci95_low_pp"], row["balanced_accuracy_ci95_high_pp"]
            interval = "pending" if low is None else f"[{low:.2f}, {high:.2f}]"
            lines.append(
                f"| {row['attention']} | {row['regime']} | {row['pattern']} | "
                f"{row['paired_subjects_completed']}/{row['subjects_expected']} | "
                f"{_format(row['balanced_accuracy_gain_pp'])} | {interval} | "
                f"{_format(row['accuracy_gain_pp'])} |"
            )
    lines += [
        "", "## Report files", "",
        "- `accuracy_by_run.csv`: each completed subject/seed/arm, repeat-averaged conditions, selected epoch, parameters and elapsed time.",
        "- `accuracy_by_subject.csv`: all conditions, seed-averaged scores and seed SD per subject.",
        "- `accuracy_table.csv`: all conditions, complete subject-averaged means and subject SD.",
        "- `robustness_table.csv`: equally weighted full/16/11/6-channel scores for each loss pattern.",
        "- `paired_tensor_gain.csv`: matched tensor-minus-baseline gains and subject bootstrap intervals.",
        "- `progress.json`: completeness, missing fits, invalid records, errors and mask-pairing issues.",
        "", "Missing-data reconstruction NRMSE is a secondary dimensionless relative feature error and is "
        "reported only when every contributing record provides a finite value. Lower NRMSE "
        "does not imply better classification. An unavailable NRMSE is left blank.", "",
    ]
    return "\n".join(lines)


def summarize(output: Path, cfg: dict, study_id: str) -> dict:
    """Refresh all reports under a process lock and return the progress record.

    Safe for independent Slurm array workers: source files must be published by
    atomic rename; this reader validates complete cell coverage before counting.
    """
    output = Path(output)
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".summary.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        declared_arms = arms(cfg)
        expected = _expected_cells(cfg)
        records = {}
        payloads = {}
        paths = {}
        valid_paths = set()
        pending = []
        invalid = []
        for subject, seed in tasks(cfg):
            for arm in declared_arms:
                path = output / "artifacts" / task_name(subject, seed) / "ARMS" / arm["name"] / "result.json"
                identity = {"subject": subject, "seed": seed, "arm": arm["name"]}
                if not path.is_file():
                    pending.append(identity)
                    continue
                try:
                    payload = _load(path)
                    indexed = _validated_result(payload, subject, seed, arm, study_id, expected)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    invalid.append({**identity, "file": str(path.relative_to(output)), "reason": str(error)})
                    continue
                key = (subject, seed, arm["name"])
                records[key] = indexed
                payloads[key] = payload
                paths[key] = path
                valid_paths.add(path.parent)
        identity_conflicts = []
        for subject, seed in tasks(cfg):
            task_keys = [key for key in records if key[:2] == (subject, seed)]
            identities = {tuple(payloads[key]["identity"][name] for name in IDENTITY_HASHES)
                          for key in task_keys}
            if len(identities) <= 1:
                continue
            # Neither the first arm encountered nor the majority identifies the
            # correct provenance. Exclude the whole conflicting task explicitly.
            identity_conflicts.append({
                "subject": subject, "seed": seed,
                "reason": "arms disagree on dataset, split or calibration identity; entire task excluded",
                "arms": [{"arm": key[2], **{name: payloads[key]["identity"][name]
                                           for name in IDENTITY_HASHES}} for key in task_keys],
            })
            for key in task_keys:
                invalid.append({"subject": subject, "seed": seed, "arm": key[2],
                                "file": str(paths[key].relative_to(output)),
                                "reason": "dataset/split/calibration mismatch across arms in this task"})
                valid_paths.discard(paths[key].parent)
                del records[key]
                del payloads[key]
        failures = []
        for path in sorted((output / "artifacts").rglob("error.json")):
            if path.parent in valid_paths:
                continue
            try:
                value = _load(path)
                if isinstance(value, dict) and value.get("study_id", study_id) != study_id:
                    continue
                failures.append({"file": str(path.relative_to(output)), "error": value})
            except (OSError, ValueError) as error:
                failures.append({"file": str(path.relative_to(output)), "error": str(error)})
        by_run = _run_table(records, payloads)
        by_subject, subject_index = _subject_tables(records, cfg, declared_arms)
        overall = _overall_tables(subject_index, cfg, declared_arms)
        robustness = _robustness_tables(subject_index, cfg, declared_arms)
        paired, pairing_issues = _paired_tables(records, cfg, study_id)
        expected_fits = len(tasks(cfg)) * len(declared_arms)
        complete = len(records) == expected_fits and not pairing_issues
        progress = {
            "study_id": study_id, "status": "complete" if complete else "incomplete",
            "completed_fits": len(records), "expected_fits": expected_fits,
            "incomplete_fits": expected_fits - len(records),
            "expected_metric_rows_per_fit": len(expected), "failed_fit_count": len(failures),
            "subjects": cfg["subjects"], "seeds": cfg["seeds"], "arms": declared_arms,
            "pending_fits": pending, "invalid_results": invalid, "failures": failures,
            "pairing_issues": pairing_issues, "identity_conflicts": identity_conflicts,
            "aggregation": "mean repeats, then seeds within each subject, then equally over all subjects",
            "primary_metric": "balanced_accuracy", "bootstrap_unit": "subject mean paired difference",
            "checkpoint_selection": "full-input validation only",
        }
        _write_csv(report / "accuracy_by_run.csv", by_run)
        _write_csv(report / "accuracy_by_subject.csv", by_subject)
        _write_csv(report / "accuracy_table.csv", overall)
        _write_csv(report / "robustness_table.csv", robustness)
        _write_csv(report / "paired_tensor_gain.csv", paired)
        _atomic_text(report / "summary.md", _summary_text(progress, overall, paired))
        # Publish progress last so it serves as a completion marker for this refresh.
        _write_json(report / "progress.json", progress)
        return progress
