"""Completeness-aware reports for saved baselines-v1 task results.

This module uses only the standard library so ``--summarize-only`` works on
machines without the scientific or GPU training stack.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .protocol import arms, tasks


METRICS = ("balanced_accuracy", "accuracy", "f1_macro")
PARTITIONS = ("validation", "test")
NEURAL_SCENARIOS = ["full", *[f"{p}_{n}" for n in (16, 11, 6)
    for p in ("random_static", "spatial_static", "dynamic_random", "dynamic_spatial")]]


def _mean(values):
    return sum(values) / len(values) if values else None


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _expected(cfg, arm):
    scenarios = ["full"] if arm["family"] == "classical" else NEURAL_SCENARIOS
    return {(partition, scenario, repeat)
            for partition in PARTITIONS for scenario in scenarios
            for repeat in range(1 if scenario == "full" else cfg["mask_repeats"])}


def _validate(value, subject, seed, arm, cfg):
    if not isinstance(value, dict) or value.get("format") != "agfl-baseline-task-result-v1":
        raise ValueError("unsupported result format")
    if value.get("status") != "complete":
        raise ValueError("result status is not complete")
    if value.get("subject") != subject or value.get("seed") != seed:
        raise ValueError("subject/seed does not match declared task path")
    if value.get("protocol") != cfg["protocol"]:
        raise ValueError("split protocol differs from requested study")
    if (value.get("arm") != arm["name"] or value.get("family") != arm["family"] or
            value.get("regime") != arm["regime"]):
        raise ValueError("arm/family differs from declared arm")
    if value.get("head_type", "flatten") != arm.get("head_type", "flatten"):
        raise ValueError("readout head differs from declared arm")
    want_scope = "full_input_only" if arm["family"] == "classical" else "full_and_degraded"
    if value.get("coverage_scope") != want_scope:
        raise ValueError("coverage scope differs from declared arm")
    if value.get("synthetic") is not False:
        raise ValueError("synthetic result cannot enter a real report")
    if value.get("selection_policy") != cfg["selection"]["policy"] and arm["family"] == "neural":
        raise ValueError("selection policy differs from requested study")
    identity = value.get("identity")
    if not isinstance(identity, dict) or identity.get("synthetic") is not False:
        raise ValueError("missing or synthetic provenance identity")
    config_path = cfg.get("config_path")
    if config_path and Path(config_path).is_file():
        expected_config = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        if identity.get("config_sha256") != expected_config:
            raise ValueError("config hash differs from requested study")
    if not _valid_hash(value.get("data_fingerprint")) or not _valid_hash(value.get("split_id")):
        raise ValueError("missing or invalid data/split fingerprint")
    if (identity.get("data_fingerprint") != value.get("data_fingerprint") or
            identity.get("split_id") != value.get("split_id")):
        raise ValueError("data/split fingerprint conflicts with identity")
    rows = value.get("metrics")
    if not isinstance(rows, list):
        raise ValueError("metrics must be a list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("metric row is not an object")
        key = (row.get("partition"), row.get("scenario"), row.get("mask_repeat"))
        if key not in _expected(cfg, arm) or key in indexed:
            raise ValueError("unexpected or duplicate metric cell")
        if not _valid_hash(row.get("mask_sha256")):
            raise ValueError("missing or malformed mask hash")
        count = row.get("sample_count")
        if type(count) is not int or count <= 0:
            raise ValueError("sample_count must be a positive integer")
        for metric in METRICS:
            score = row.get(metric)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid {metric}")
        recalls = row.get("per_class_recall")
        if (not isinstance(recalls, list) or len(recalls) != 4 or
                any(isinstance(score, bool) or not isinstance(score, (int, float)) or
                    not math.isfinite(score) or not 0 <= score <= 1 for score in recalls)):
            raise ValueError("per_class_recall must contain four finite class recalls")
        indexed[key] = row
    expected = _expected(cfg, arm)
    if set(indexed) != expected:
        raise ValueError(f"missing metric cells: {len(expected - set(indexed))}")
    return indexed


def _score(indexed, partition, scenario, metric):
    rows = [row for (part, sc, _), row in indexed.items() if part == partition and sc == scenario]
    return _mean([float(row[metric]) for row in rows])


def _dataset_identity(task_dir: Path, value: dict, subject: int, seed: int) -> str:
    """Verify the prepared-data identity; return its seed-independent dataset hash.

    data_fingerprint includes the split and train-fitted normalization, so it
    must only match across arms of the SAME task. dataset_fingerprint identifies
    the underlying loaded signals and is the correct comparison across seeds.
    The persisted metadata ties the two together without changing saved results.
    """
    dataset = json.loads((task_dir / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("dataset.json must contain a provenance object")
    provenance = dataset.get("provenance")
    raw_hash = dataset.get("dataset_fingerprint")
    if not isinstance(provenance, dict) or not _valid_hash(raw_hash):
        raise ValueError("dataset.json lacks a valid underlying dataset fingerprint")
    if (provenance.get("dataset_fingerprint") != raw_hash or
            provenance.get("subject") != subject or provenance.get("seed") != seed or
            provenance.get("split_id") != value["split_id"]):
        raise ValueError("dataset.json provenance conflicts with task/result identity")
    config = provenance.get("config")
    if not isinstance(config, dict) or config.get("config_sha256") != value["identity"].get("config_sha256"):
        raise ValueError("dataset.json config conflicts with result identity")
    if not isinstance(dataset.get("normalization_stats"), dict) or not isinstance(dataset.get("sample_ids"), list):
        raise ValueError("dataset.json lacks normalization or sample IDs")
    payload = {"provenance": provenance, "normalization": dataset["normalization_stats"],
               "sample_ids": dataset["sample_ids"]}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if actual != dataset.get("data_fingerprint") or actual != value["data_fingerprint"]:
        raise ValueError("prepared data_fingerprint does not match dataset.json contents")
    return raw_hash


def summarize(cfg) -> dict:
    """Read records under ``cfg['output_dir']`` and rebuild CSV/Markdown reports.

    The cohort is complete only when every declared task and arm has one valid
    record. The returned dictionary is also written as ``report/progress.json``.
    """
    output = Path(cfg["output_dir"])
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    declared_arms = arms(cfg)
    valid, invalid, missing, failures = {}, [], [], []
    identities = {}
    subject_data = {}
    for subject, seed in tasks(cfg):
        task = f"A{subject:02d}_seed_{seed}"
        for arm in declared_arms:
            arm_dir = output / "artifacts" / task / "ARMS" / arm["name"]
            path = arm_dir / "result.json"
            key = (subject, seed, arm["name"])
            if not path.exists():
                missing.append({"subject": subject, "seed": seed, "arm": arm["name"], "reason": "result missing"})
                failure = arm_dir / "failure.json"
                if failure.exists():
                    try:
                        failures.append({**json.loads(failure.read_text()), "subject": subject, "seed": seed})
                    except (OSError, ValueError):
                        failures.append({"subject": subject, "seed": seed, "arm": arm["name"], "reason": "corrupt failure record"})
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                indexed = _validate(value, subject, seed, arm, cfg)
                identity = value["identity"]
                # Study-wide provenance fields cannot vary across tasks. Per-task data/split hashes can.
                global_identity = {k: v for k, v in identity.items() if k not in ("data_fingerprint", "split_id", "subject", "seed")}
                prior = identities.get("global")
                if prior is not None and prior != global_identity:
                    raise ValueError("incompatible source/config/selection/budget/preprocessing identity")
                identities["global"] = global_identity
                raw_hash = _dataset_identity(arm_dir.parent.parent, value, subject, seed)
                data_key = subject
                old_data = subject_data.get(data_key)
                if old_data is not None and old_data != raw_hash:
                    raise ValueError("underlying dataset fingerprint differs across seeds for this participant")
                subject_data[data_key] = raw_hash
                identities[key] = identity
                valid[key] = (value, indexed)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as error:
                invalid.append({"subject": subject, "seed": seed, "arm": arm["name"],
                                "path": str(path), "reason": str(error)})

    run_rows = []
    for (subject, seed, arm_name), (value, indexed) in valid.items():
        arm = next(item for item in declared_arms if item["name"] == arm_name)
        for partition in PARTITIONS:
            for scenario in (["full"] if arm["family"] == "classical" else NEURAL_SCENARIOS):
                entries = [row for (part, sc, _), row in indexed.items() if part == partition and sc == scenario]
                row = {"subject": subject, "seed": seed, "arm": arm_name, "family": arm["family"],
                       "head_type": value.get("head_type", "flatten"),
                       "protocol": value["protocol"], "selection_policy": value["selection_policy"],
                       "partition": partition, "scenario": scenario, "mask_repeats": len(entries),
                       "sample_count": entries[0]["sample_count"],
                       "data_fingerprint": value["data_fingerprint"], "split_id": value["split_id"],
                       "config_sha256": value["identity"].get("config_sha256"),
                       "source_sha256": json.dumps(value["identity"].get("source_sha256", {}), sort_keys=True),
                       "mask_hashes": ";".join(item["mask_sha256"] for item in entries)}
                for metric in METRICS:
                    row[metric] = _score(indexed, partition, scenario, metric)
                for class_index in range(4):
                    row[f"recall_class_{class_index}"] = _mean([
                        float(item["per_class_recall"][class_index]) for item in entries])
                run_rows.append(row)

    participant_rows, participant_index = [], {}
    for arm in declared_arms:
        for subject in cfg["subjects"]:
            for partition in PARTITIONS:
                scenarios = ["full"] if arm["family"] == "classical" else NEURAL_SCENARIOS
                seed_values = [(seed, valid.get((subject, seed, arm["name"]))) for seed in cfg["seeds"]]
                complete = all(record is not None for _, record in seed_values)
                for scenario in scenarios:
                    row = {"subject": subject, "arm": arm["name"], "family": arm["family"],
                           "head_type": arm.get("head_type", "flatten"),
                           "partition": partition, "scenario": scenario, "complete": complete,
                           "seeds_completed": sum(record is not None for _, record in seed_values),
                           "seeds_expected": len(cfg["seeds"])}
                    for metric in METRICS:
                        scores = [_score(record[1], partition, scenario, metric) for _, record in seed_values if record]
                        row[metric] = _mean(scores) if complete else None
                    for metric in METRICS:
                        degradations = [_score(record[1], partition, "full", metric) - _score(record[1], partition, scenario, metric)
                                        for _, record in seed_values if record is not None and scenario != "full"]
                        row[f"{metric}_degradation"] = _mean(degradations) if complete and scenario != "full" else (0.0 if complete else None)
                    participant_rows.append(row)
                    participant_index[(subject, arm["name"], partition, scenario)] = row

    cohort_rows, degraded_rows, degradation_rows = [], [], []
    for arm in declared_arms:
        scenarios = ["full"] if arm["family"] == "classical" else NEURAL_SCENARIOS
        for partition in PARTITIONS:
            for scenario in scenarios:
                people = [participant_index[(subject, arm["name"], partition, scenario)] for subject in cfg["subjects"]]
                complete = all(row["complete"] for row in people)
                row = {"arm": arm["name"], "family": arm["family"], "partition": partition,
                       "head_type": arm.get("head_type", "flatten"),
                       "scenario": scenario, "complete": complete,
                       "subjects_completed": sum(person["complete"] for person in people),
                       "subjects_expected": len(cfg["subjects"])}
                for metric in METRICS:
                    row[metric] = _mean([person[metric] for person in people]) if complete else None
                    row[f"{metric}_degradation"] = _mean([person[f"{metric}_degradation"] for person in people]) if complete else None
                cohort_rows.append(row)
                if scenario != "full":
                    degraded_rows.append(row.copy())
                    degradation_rows.append({k: row[k] for k in ("arm", "family", "partition", "scenario", "complete", "subjects_completed", "subjects_expected", *[f"{m}_degradation" for m in METRICS])})

    paired_rows, pairing_issues = [], []
    neural = [arm for arm in declared_arms if arm["family"] == "neural"]
    for ai, left in enumerate(neural):
        for right in neural[ai + 1:]:
            for subject, seed in tasks(cfg):
                lk, rk = (subject, seed, left["name"]), (subject, seed, right["name"])
                if lk not in valid or rk not in valid:
                    continue
                lv, li = valid[lk]; rv, ri = valid[rk]
                reasons = []
                for field in ("data_fingerprint", "split_id"):
                    if lv.get(field) != rv.get(field): reasons.append(field + " mismatch")
                if lv.get("protocol") != rv.get("protocol") or lv.get("selection_policy") != rv.get("selection_policy"):
                    reasons.append("protocol/selection mismatch")
                mismatched = [key for key in li if li[key]["mask_sha256"] != ri[key]["mask_sha256"]]
                if mismatched: reasons.append("mask hash mismatch")
                if reasons:
                    pairing_issues.append({"subject": subject, "seed": seed, "left_arm": left["name"],
                        "right_arm": right["name"], "reason": "; ".join(reasons), "mismatched_cells": len(mismatched)})
                    continue
                for partition in PARTITIONS:
                    for scenario in NEURAL_SCENARIOS:
                        row = {"subject": subject, "seed": seed, "left_arm": left["name"],
                               "right_arm": right["name"], "partition": partition, "scenario": scenario,
                               "left_head_type": left.get("head_type", "flatten"),
                               "right_head_type": right.get("head_type", "flatten"),
                               "data_fingerprint": lv["data_fingerprint"], "split_id": lv["split_id"],
                               "mask_hashes": ";".join(li[k]["mask_sha256"] for k in sorted(li) if k[0] == partition and k[1] == scenario)}
                        for metric in METRICS:
                            row[metric + "_difference"] = _score(ri, partition, scenario, metric) - _score(li, partition, scenario, metric)
                        paired_rows.append(row)

    # Add paired cohort means after repeat-averaging each run, seed-averaging
    # each participant, and weighting participants equally.
    paired_groups = {}
    for row in paired_rows:
        paired_groups.setdefault((row["left_arm"], row["right_arm"], row["partition"], row["scenario"]), []).append(row)
    for (left_name, right_name, partition, scenario), rows in paired_groups.items():
        by_subject = {}
        for row in rows:
            by_subject.setdefault(row["subject"], []).append(row)
        complete_subjects = [subject for subject in cfg["subjects"]
                             if len(by_subject.get(subject, [])) == len(cfg["seeds"])]
        complete = len(complete_subjects) == len(cfg["subjects"])
        aggregate = {"scope": "cohort", "subject": "", "seed": "", "left_arm": left_name,
                     "right_arm": right_name, "partition": partition, "scenario": scenario,
                     "complete": complete, "subjects_completed": len(complete_subjects),
                     "subjects_expected": len(cfg["subjects"])}
        for metric in METRICS:
            key = metric + "_difference"
            subject_means = [_mean([row[key] for row in by_subject[subject]]) for subject in complete_subjects]
            aggregate[key] = _mean(subject_means) if complete else None
        paired_rows.append(aggregate)

    expected_fits = len(tasks(cfg)) * len(declared_arms)
    progress = {"format": "agfl-baseline-report-v1", "protocol": cfg["protocol"], "synthetic": False,
        "expected_fits": expected_fits, "completed_fits": len(valid), "missing": missing,
        "invalid": invalid, "failures": failures, "pairing_issues": pairing_issues,
        "complete": len(valid) == expected_fits and not invalid and not missing}
    for filename, rows in (("accuracy_by_run.csv", run_rows), ("accuracy_by_subject.csv", participant_rows),
        ("accuracy_table.csv", cohort_rows), ("degraded_table.csv", degraded_rows),
        ("full_input_degradation.csv", degradation_rows), ("paired_neural_differences.csv", paired_rows),
        ("pairing_issues.csv", pairing_issues), ("invalid_results.csv", invalid), ("missing_results.csv", missing)):
        _csv(report / filename, rows)
    _json(report / "progress.json", progress)
    lines = ["# Baseline study report", "", f"Protocol: `{cfg['protocol']}`. Synthetic results excluded: yes.",
        f"", f"**{len(valid)}/{expected_fits} valid task-arm records.** Cohort complete: {progress['complete']}.",
        f"Missing: {len(missing)}; invalid/corrupt/incompatible: {len(invalid)}; pairing exclusions: {len(pairing_issues)}.", "",
        "Scores are averaged over mask repeats, then seeds within each participant, then participants equally. Trial counts do not weight participants.",
        "Across seeds, underlying dataset identities must match; split-specific normalization fingerprints are verified against each task's dataset.json and may differ.",
        "Cohort values remain blank until every declared seed for every participant is present and valid. The classical covariance arm contributes full-input rows only.",
        "Paired neural differences are right arm minus left arm and are emitted only when subject, seed, data/split identities, protocol, selection policy, and exact condition mask hashes match.", "",
        "## Full-input cohort scores", "", "| Arm | Partition | Complete | Subjects | Balanced accuracy | Accuracy | Macro-F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in cohort_rows:
        if row["scenario"] == "full":
            val = lambda name: "pending" if row[name] is None else f"{row[name]:.4f}"
            lines.append(f"| {row['arm']} | {row['partition']} | {row['complete']} | {row['subjects_completed']}/{row['subjects_expected']} | {val('balanced_accuracy')} | {val('accuracy')} | {val('f1_macro')} |")
    lines += ["", "## Report files", "", "- `accuracy_by_run.csv`: valid run/seed/arm scores with repeat-averaged conditions and provenance.",
        "- `accuracy_by_subject.csv`: seed means per participant, with completion counts.", "- `accuracy_table.csv` and `degraded_table.csv`: equally weighted cohort scores.",
        "- `full_input_degradation.csv`: full minus degraded scores, in score units.", "- `paired_neural_differences.csv`: right-minus-left paired neural contrasts.",
        "- `pairing_issues.csv`, `invalid_results.csv`, `missing_results.csv`, `progress.json`: audit trail.", ""]
    (report / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return progress
