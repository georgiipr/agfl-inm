"""Checksum-verified task collection and hierarchical descriptive summaries."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

from .protocol import audit_identity, digest, file_sha256, tasks
from .study import CHECK_NAMES, _atomic_json


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read audit JSON {path}: {error}") from error


def _write_task_fixture(root, identity, subject, seed, synthetic):
    """Small integrity fixture shared by smoke and report validation tests."""
    directory = root / "tasks" / f"A{subject:02d}_seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    probe_file = directory / "probes" / "fixture.npz"
    probe_file.parent.mkdir(parents=True, exist_ok=True)
    # A deterministic numeric payload with no pickle and no research observations.
    probe_file.write_bytes(b"synthetic-fixture-numeric-probe-v1\n")
    checks = {name: True for name in CHECK_NAMES}
    task = {"schema_name": "agfl-encoder-audit-task-v1", "subject": subject, "seed": seed,
        "status": "complete", "synthetic": synthetic, "partitions": ["train", "validation"],
        "input_study_ids": {"baseline": "0" * 64, "legacy": "1" * 64},
        "audit_id": identity["audit_id"], "config_sha256": identity["config_sha256"],
        "source_sha256": identity["source_sha256"], "source_files_sha256": identity["source_files_sha256"],
        "packages": identity["packages"], "input_checksums": {}, "input_manifest_sha256": {},
        "split_ids": {}, "alignment": {"status": "synthetic_fixture"},
        "clean_checkpoints": {"status": "synthetic_fixture"},
        "probes": {"status": "synthetic_fixture", "probes": {}},
        "reconstruction": {"status": "synthetic_fixture", "conditions": []},
        "checks": checks, "errors": [], "timings": {"elapsed_seconds": 0.0},
        "tolerances": {}, "artifact_sha256": {"probes/fixture.npz": file_sha256(probe_file)}}
    _atomic_json(directory / "audit.json", task)
    return task


def _task_records(cfg, identity):
    root = Path(cfg["output_dir"])
    configured = tasks(cfg)
    if len(set(configured)) != len(configured):
        raise ValueError("Duplicate configured task")
    records, contents, failed, seen = [], [], [], set()
    for subject, seed in configured:
        seen.add((subject, seed))
        path = root / "tasks" / f"A{subject:02d}_seed_{seed}" / "audit.json"
        if not path.is_file():
            continue
        task = _read(path)
        if task.get("subject") != subject or task.get("seed") != seed:
            raise ValueError(f"Task identity mismatch: {path}")
        if task.get("audit_id") != identity["audit_id"] or task.get("config_sha256") != identity["config_sha256"]:
            raise ValueError(f"Task config/source identity mismatch: {path}")
        if task.get("source_files_sha256") != identity["source_files_sha256"]:
            raise ValueError(f"Source files changed since task: {path}")
        if task.get("synthetic") is not False:
            raise ValueError(f"Synthetic/real task mixing is forbidden: {path}")
        if task.get("partitions") != ["train", "validation"]:
            raise ValueError(f"Unsafe task partitions: {path}")
        if not isinstance(task.get("checks"), dict) or set(task["checks"]) != set(CHECK_NAMES) or not all(
                type(v) is bool for v in task["checks"].values()):
            raise ValueError(f"Task check schema is invalid: {path}")
        for rel, expected in task.get("artifact_sha256", {}).items():
            artifact = (path.parent / rel).resolve()
            if not artifact.is_relative_to(path.parent.resolve()) or not artifact.is_file() or file_sha256(artifact) != expected:
                raise ValueError(f"Task artifact checksum mismatch: {artifact}")
        if task.get("status") != "complete":
            if task.get("status") != "failed":
                raise ValueError(f"Unknown task status: {path}")
            failed.append({"subject": subject, "seed": seed,
                "path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                "errors": task.get("errors", [])})
            continue
        if not all(task["checks"].values()):
            raise ValueError(f"Task acceptance checks failed: {path}")
        records.append({"subject": subject, "seed": seed,
            "path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)})
        contents.append(task)
    # Unknown on-disk tasks indicate possible identity reuse or duplicated rows.
    for path in (root / "tasks").glob("A*_seed_*/audit.json") if (root / "tasks").exists() else ():
        task = _read(path)
        if (task.get("subject"), task.get("seed")) not in set(tasks(cfg)):
            raise ValueError(f"Unexpected task artifact: {path}")
    return records, contents, failed


def _hierarchy(cfg, contents):
    """Average repeats, then seeds, then participants; never weight by trial count."""
    import math
    by_condition = {}
    for task in contents:
        subject, seed = task["subject"], task["seed"]
        for row in task.get("reconstruction", {}).get("conditions", []):
            key = (row.get("condition"), row.get("pattern"), row.get("retained"))
            bucket = by_condition.setdefault(key, {}).setdefault(subject, {}).setdefault(seed, [])
            bucket.append(row)
    rows = []
    expected_subjects = set(cfg["subjects"])
    expected_seeds = set(cfg["seeds"])
    for (name, pattern, retained), subject_map in sorted(by_condition.items(), key=lambda item: str(item[0])):
        participant = {}
        seed_counts = {}
        for subject, seed_map in subject_map.items():
            per_seed = {}
            for seed, repeats in seed_map.items():
                seed_counts[f"{subject}:{seed}"] = len(repeats)
                # Mask repeats are summarized within a seed. Keep metrics/views separate.
                metrics = {}
                for view in ("zero_fill", "tucker_completion", "full_features"):
                    keys = sorted({metric for rep in repeats for metric in rep.get("metrics", {}).get(view, {})
                                   if metric in ("balanced_accuracy", "log_loss")})
                    metrics[view] = {metric: _mean([rep["metrics"][view][metric] for rep in repeats
                        if rep.get("metrics", {}).get(view, {}).get(metric) is not None]) for metric in keys}
                per_seed[seed] = metrics
            participant[subject] = per_seed
        complete = set(subject_map) == expected_subjects and all(
            set(subject_map[s]) == expected_seeds for s in expected_subjects)
        views = {}
        for view in ("zero_fill", "tucker_completion", "full_features"):
            views[view] = {}
            metric_names = sorted({m for seed_map in participant.values() for metrics in seed_map.values()
                                   for m in metrics.get(view, {})})
            for metric in metric_names:
                participant_values = []
                for subject in expected_subjects:
                    vals = [participant.get(subject, {}).get(seed, {}).get(view, {}).get(metric)
                            for seed in expected_seeds]
                    vals = [x for x in vals if x is not None]
                    if vals:
                        participant_values.append(_mean(vals))
                views[view][metric] = _mean(participant_values) if complete and len(participant_values) == len(expected_subjects) else None
        rows.append({"condition": name, "pattern": pattern, "retained": retained,
            "status": "complete" if complete else "partial", "participants_available": len(subject_map),
            "participants_expected": len(expected_subjects), "views": views,
            "participant_seed_repeat_counts": seed_counts,
            "cohort_mean_available": complete})
    return rows


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def summarize(cfg):
    """Validate task receipts and atomically write partial/complete report views."""
    root = Path(cfg["output_dir"])
    for key in ("baseline_dir", "legacy_dir", "data_dir"):
        source = Path(cfg[key]).resolve()
        if root.resolve() == source or root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
            raise ValueError(f"output_dir overlaps {key}")
    identity = audit_identity(cfg)
    records, contents, failed = _task_records(cfg, identity)
    expected = len(tasks(cfg))
    complete = len(records) == expected and bool(contents)
    input_ids = None
    for task in contents:
        ids = task["input_study_ids"]
        if input_ids is None: input_ids = ids
        elif ids != input_ids: raise ValueError("Input study identities differ across tasks")
    report = {"schema_name": "agfl-encoder-audit-report-v1", "status": "complete" if complete else "partial",
        "synthetic": False, "partitions": ["train", "validation"], "subjects": cfg["subjects"],
        "seeds": cfg["seeds"], "input_study_ids": input_ids or {"baseline": None, "legacy": None},
        "source_files_sha256": identity["source_files_sha256"], "tasks": records,
        "aggregation": _hierarchy(cfg, contents), "unavailable_comparisons": _unavailable(contents),
        "failed_tasks": failed,
        "audit_id": identity["audit_id"], "config_sha256": identity["config_sha256"]}
    report_dir = root / "report"
    evidence = report_dir / "evidence.json"
    _atomic_json(evidence, report)
    md = ["# Encoder audit summary", "", f"Status: **{report['status']}**; tasks: {len(records)}/{expected}.",
          "", "Validation is reused and exploratory. Cohort means are withheld unless every configured task is present.", "",
          "| Condition | Retained | Status | Participants | Completion BA | Zero-fill BA |", "|---|---:|---|---:|---:|---:|"]
    for row in report["aggregation"]:
        views = row["views"]
        md.append(f"| {row['condition']} | {row['retained']} | {row['status']} | {row['participants_available']}/{row['participants_expected']} | {views['tucker_completion'].get('balanced_accuracy')} | {views['zero_fill'].get('balanced_accuracy')} |")
    _atomic_text(report_dir / "summary.md", "\n".join(md) + "\n")
    columns = ("condition", "pattern", "retained", "status", "participants_available", "participants_expected",
               "completion_balanced_accuracy", "zero_fill_balanced_accuracy")
    csv_path = report_dir / "summary.csv"
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns); writer.writeheader()
    for row in report["aggregation"]:
        writer.writerow({"condition": row["condition"], "pattern": row["pattern"], "retained": row["retained"],
            "status": row["status"], "participants_available": row["participants_available"],
            "participants_expected": row["participants_expected"],
            "completion_balanced_accuracy": row["views"]["tucker_completion"].get("balanced_accuracy"),
            "zero_fill_balanced_accuracy": row["views"]["zero_fill"].get("balanced_accuracy")})
    _atomic_text(csv_path, buffer.getvalue())
    return {"status": report["status"], "paths": {"evidence": str(evidence),
        "markdown": str(report_dir / "summary.md"), "csv": str(csv_path)}, "task_count": len(records)}


def _unavailable(contents):
    rows = []
    for task in contents:
        pairing = task.get("alignment", {}).get("cross_study_pairing", {})
        if pairing.get("status") != "matched":
            rows.append({"subject": task["subject"], "seed": task["seed"],
                "comparison": "cross_study_descriptive_pairing", "status": pairing.get("status", "unavailable"),
                "details": pairing})
    return rows
