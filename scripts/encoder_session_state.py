#!/usr/bin/env python3
"""Stdlib-only prompt, receipt, and independent acceptance checks for sessions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan(root):
    directory = root / "plans/encoder-audit"
    manifest = read_json(directory / "manifest.json")
    ids = [s["id"] for s in manifest["sessions"]]
    if not ids or ids != [f"{i:02d}" for i in range(1, len(ids) + 1)]:
        raise ValueError("Session IDs must be consecutive, ordered two-digit strings")
    if manifest["default_through"] not in ids:
        raise ValueError("Invalid default_through")
    for session in manifest["sessions"]:
        relative = Path(session["prompt"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Session prompts must be relative to the plan directory")
        if not (directory / relative).is_file():
            raise ValueError(f"Missing prompt: {relative}")
    return directory, manifest


def selected(root, sid):
    directory, manifest = plan(root)
    for session in manifest["sessions"]:
        if session["id"] == sid:
            return directory, session
    raise ValueError(f"Unknown session: {sid}")


def fingerprint(root, sid):
    directory, session = selected(root, sid)
    paths = sorted(path for path in directory.rglob("*") if path.is_file()
                   and path.suffix in {".md", ".json"})
    paths += [root / "scripts" / name for name in
              ("run_encoder_sessions.sh", "encoder_session_state.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def validate_report(path, sid):
    report = read_json(path)
    required = {"session_id", "status", "summary", "files_changed", "checks",
                "blockers", "next_session_notes"}
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("Final response must contain exactly the declared JSON fields")
    for key in ("session_id", "status", "summary", "next_session_notes"):
        if not isinstance(report[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("files_changed", "checks", "blockers"):
        if not isinstance(report[key], list) or any(not isinstance(x, str) for x in report[key]):
            raise ValueError(f"{key} must be an array of strings")
    if report["session_id"] != sid:
        raise ValueError(f"Wrong session ID: expected {sid}, received {report['session_id']}")
    if report["status"] != "completed":
        raise ValueError(f"Session {sid} did not complete: {report['status']}; "
                         + "; ".join(report["blockers"]) + " " + report["summary"])
    if report["blockers"]:
        raise ValueError(f"Session {sid} reports completed but blockers is nonempty: "
                         + "; ".join(report["blockers"])
                         + ". Unfinished session work must be reported as blocked. "
                         "Downstream input-readiness findings belong in the inventory and "
                         "next_session_notes only when this session's work and checks passed.")
    if not report["summary"].strip():
        raise ValueError("Completed sessions need a nonempty summary")
    if not report["checks"] or any(not check.strip() for check in report["checks"]):
        raise ValueError("Completed sessions need nonempty descriptions of actual checks")
    for name in report["files_changed"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Changed-file paths must be relative to the workspace")
    return report


def receipt_path(state, sid):
    return state / "completed" / f"{sid}.json"


def check_receipt(root, state, sid):
    record = read_json(receipt_path(state, sid))
    if (record.get("workspace") != str(root)
            or record.get("session_id") != sid
            or record.get("fingerprint") != fingerprint(root, sid)):
        raise ValueError(f"Session {sid} receipt is stale or belongs to another checkout. "
                         "Review the changed plans and use a new --state-dir.")
    validate_report(record["report_path"], sid)
    if record.get("report_sha256") != hashlib.sha256(Path(record["report_path"]).read_bytes()).hexdigest():
        raise ValueError(f"Session {sid} handoff was modified after verification")
    return record


def prompt(root, state, sid):
    directory, session = selected(root, sid)
    print(f"Execute encoder investigation session {sid}: {session['title']}.\n")
    print((directory / "COMMON.md").read_text())
    print("\n## Shared scientific and interface contract\n")
    print((directory / "CONTRACT.md").read_text())
    print("\n## Prior verified handoffs (context, not additional tasks)\n")
    for previous in plan(root)[1]["sessions"]:
        if previous["id"] >= sid:
            break
        record = check_receipt(root, state, previous["id"])
        handoff = read_json(record["report_path"])
        print(json.dumps({k: handoff[k] for k in
                          ("session_id", "summary", "next_session_notes")}, ensure_ascii=False))
    print("\n## Current task\n")
    print((directory / session["prompt"]).read_text())
    print("\nRequired artifacts:")
    print("\n".join(session["artifacts"]))
    print("Required unittest patterns: " + ", ".join(session["tests"]))
    print(f"Use session_id {sid!r} in your final structured JSON response.")


def gate(root, sid):
    """Keep the evidence-review session behind a complete real audit, not prose."""
    _, session = selected(root, sid)
    if "evidence" not in session:
        return
    evidence_path = (root / session["evidence"]).resolve()
    if not evidence_path.is_relative_to(root):
        raise ValueError("Evidence must be inside this workspace")
    report = read_json(evidence_path)
    if (not isinstance(report, dict)
            or report.get("schema_name") != "agfl-encoder-audit-report-v1"
            or report.get("status") != "complete" or report.get("synthetic") is not False
            or report.get("partitions") != ["train", "validation"]
            or report.get("subjects") != list(range(1, 10))
            or report.get("seeds") != [0, 1, 2]):
        raise ValueError("Evidence requires a complete real 9-subject/3-seed train/validation audit")
    if any(type(v) is not int for v in report["subjects"] + report["seeds"]):
        raise ValueError("Subject and seed identities must be integers")
    identities = report.get("input_study_ids", {})
    if set(identities) != {"baseline", "legacy"} or not all(
            isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in identities.values()):
        raise ValueError("Evidence lacks baseline/legacy study identities")
    scientific_files = sorted([*root.glob("*.py"), *(root / "agfl").rglob("*.py"),
                               *(root / "inm").rglob("*.py")])
    expected_source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in scientific_files}
    if not expected_source or report.get("source_files_sha256") != expected_source:
        raise ValueError("Audit source changed or provenance is incomplete; regenerate into a fresh output")
    expected_tasks = {(s, seed) for s in range(1, 10) for seed in range(3)}
    tasks = report.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) != len(expected_tasks):
        raise ValueError("Evidence must contain exactly 27 task records")
    seen, paths = set(), set()
    output = evidence_path.parent.parent
    required_checks = {"source_integrity", "trial_alignment", "split_isolation",
                       "normalization_replay", "checkpoint_replay", "probe_controls", "mask_pairing"}
    for entry in tasks:
        if not isinstance(entry, dict) or any(type(entry.get(k)) is not int for k in ("subject", "seed")):
            raise ValueError("Each task needs integer subject and seed identities")
        key = (entry.get("subject"), entry.get("seed"))
        if key not in expected_tasks or key in seen:
            raise ValueError("Duplicate or unexpected evidence task")
        seen.add(key)
        relative = Path(entry["path"])
        path = (output / relative).resolve()
        if (relative.is_absolute() or not path.is_relative_to(output) or path in paths
                or path == evidence_path):
            raise ValueError("Task evidence path must be unique and inside the audit output")
        paths.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {relative}")
        task = read_json(path)
        if (not isinstance(task, dict)
                or any(type(task.get(k)) is not int for k in ("subject", "seed"))):
            raise ValueError("Invalid task identity types")
        if (task.get("subject"), task.get("seed")) != key or (
                task.get("status") != "complete" or task.get("synthetic") is not False
                or task.get("partitions") != ["train", "validation"]
                or task.get("input_study_ids") != identities):
            raise ValueError(f"Invalid task evidence identity: {relative}")
        checks = task.get("checks", {})
        if set(checks) != required_checks or any(v is not True for v in checks.values()):
            raise ValueError(f"Unresolved integrity or diagnostic checks: {relative}")
        for section in ("alignment", "clean_checkpoints", "probes", "reconstruction"):
            if not isinstance(task.get(section), dict) or not task[section]:
                raise ValueError(f"Missing diagnostic section {section}: {relative}")
    print(f"Session {sid}: complete real audit evidence and task checksums verified")


def verify(root, sid):
    _, session = selected(root, sid)
    gate(root, sid)
    for name in session["artifacts"]:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Required nonempty artifact missing or outside workspace: {name}")
    subprocess.run(["git", "diff", "--check"], cwd=root, check=True)
    legacy = subprocess.run([sys.executable, "run.py", "--config", "configs/study.json", "--plan"],
                            cwd=root, check=True, text=True, capture_output=True).stdout
    if ("Total classifier fits: 378" not in legacy
            or len(re.findall(r"^\d+: A\d+, seed=\d+, classifier fits=14$", legacy, re.M)) != 27):
        raise ValueError("Legacy experiment no longer plans 27 tasks / 378 fits")
    sys.path.insert(0, str(root))
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for pattern in session["tests"]:
        discovered = loader.discover(str(root / "tests/encoder_audit"), pattern=pattern)
        if discovered.countTestCases() == 0:
            raise ValueError(f"No acceptance tests discovered for {pattern}")
        suite.addTests(discovered)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise ValueError("Acceptance tests failed or were skipped; session is incomplete")
    if session.get("baseline_regression"):
        subprocess.run([sys.executable, "-c", (
            "import unittest, sys; "
            "suite=unittest.defaultTestLoader.discover('tests/baselines', pattern='test_*.py'); "
            "assert suite.countTestCases() > 0, 'No baseline regression tests'; "
            "result=unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)"
        )], cwd=root, check=True)
    print(f"Session {sid}: {result.testsRun} tests passed; artifacts and legacy planner verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "ids", "default", "prompt", "fingerprint",
                                         "validate-report", "verify", "receipt", "check-receipt", "gate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--effort")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action in {"list", "ids", "default"}:
        manifest = plan(root)[1]
        if args.action == "default":
            print(manifest["default_through"])
        else:
            for item in manifest["sessions"]:
                print(item["id"] if args.action == "ids" else f"{item['id']}  {item['title']}")
        return
    selected(root, args.session)
    if args.action == "fingerprint":
        print(fingerprint(root, args.session))
    elif args.action == "prompt":
        prompt(root, args.state.resolve(), args.session)
    elif args.action == "validate-report":
        validate_report(args.report, args.session)
    elif args.action == "gate":
        gate(root, args.session)
    elif args.action == "verify":
        verify(root, args.session)
    elif args.action == "check-receipt":
        check_receipt(root, args.state.resolve(), args.session)
    elif args.action == "receipt":
        validate_report(args.report, args.session)
        path = receipt_path(args.state.resolve(), args.session)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"session_id": args.session, "workspace": str(root),
                  "fingerprint": fingerprint(root, args.session),
                  "report_path": str(args.report.resolve()),
                  "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  "model": args.model, "effort": args.effort,
                  "verified_at": datetime.now(timezone.utc).isoformat()}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Session check failed: {error}", file=sys.stderr)
        sys.exit(1)
