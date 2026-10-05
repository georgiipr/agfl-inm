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
    directory = root / "plans/candidate-followup"
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
              ("run_followup_sessions.sh", "followup_session_state.py", "run_candidate_experiments.sh")]
    paths += [root / "plans/encoder-candidates/CONTRACT.md"]
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
    print(f"Execute candidate follow-up session {sid}: {session['title']}.\n")
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


ARMS = ('local_control', 'local_power', 'spatial_eegnet',
        'spatial_filterbank', 'spatial_transformer')
CHECKS = {'split_isolation', 'normalization_train_only', 'raw_masking',
          'checkpoint_reload', 'probe_reload', 'mask_pairing', 'identities'}
CONDITIONS = {'full_22': 1, 'random_static_16': 5, 'dynamic_random_16': 5,
              'random_static_6': 5, 'dynamic_random_6': 5}


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def _checked_path(base, relative, expected, label):
    if not isinstance(relative, str) or not relative or not _hash(expected):
        raise ValueError(f'Malformed {label} path/checksum')
    rel = Path(relative)
    path = (base / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(base.resolve()):
        raise ValueError(f'{label} path must be inside its output directory')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'{label} checksum mismatch: {relative}')
    return path


def _metrics(value, label):
    import math
    if not isinstance(value, dict):
        raise ValueError(f'Missing {label} metrics')
    for key in ('balanced_accuracy', 'log_loss'):
        number = value.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError(f'Invalid {label} {key}')
        if key == 'balanced_accuracy' and number > 1:
            raise ValueError(f'Invalid {label} balanced_accuracy')


def gate(root, sid):
    """Review only complete real, paired, artifact-verified candidate evidence."""
    _, session = selected(root, sid)
    if 'evidence' not in session:
        return
    evidence_path = (root / session['evidence']).resolve()
    if not evidence_path.is_relative_to(root):
        raise ValueError('Evidence must be inside this workspace')
    pilot = session.get('pilot', False)
    if pilot:
        task_path = evidence_path.parent.parent / 'tasks/A01_seed_0/task.json'
        task = read_json(task_path)
        # An in-memory validation envelope only; never write a complete-cohort report.
        report = {k: task.get(k) for k in ('status', 'synthetic', 'partitions',
            'training_regime', 'study_id', 'config_sha256', 'packages', 'source_files_sha256')}
        report.update(schema_name='agfl-encoder-candidates-report-v1',
            subjects=list(range(1, 10)), seeds=[0, 1, 2], arms=list(ARMS),
            tasks=[{'subject': 1, 'seed': 0, 'path': 'tasks/A01_seed_0/task.json',
                    'sha256': hashlib.sha256(task_path.read_bytes()).hexdigest()}])
    else:
        report = read_json(evidence_path)
    if (not isinstance(report, dict)
            or report.get('schema_name') != 'agfl-encoder-candidates-report-v1'
            or report.get('status') != 'complete' or report.get('synthetic') is not False
            or report.get('partitions') != ['train', 'validation']
            or report.get('training_regime') != 'full'
            or report.get('subjects') != list(range(1, 10))
            or report.get('seeds') != [0, 1, 2] or report.get('arms') != list(ARMS)):
        raise ValueError('Evidence requires a complete real 9-subject/3-seed/five-arm train/validation study')
    if any(type(v) is not int for v in report['subjects'] + report['seeds']):
        raise ValueError('Subject and seed identities must be integers')
    if not _hash(report.get('study_id')):
        raise ValueError('Evidence lacks a valid study identity')
    config_path = root / 'configs/encoder-candidates-reproducible.json'
    if report.get('config_sha256') != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError('Candidate config changed; use a fresh study output')
    packages = report.get('packages')
    if (not isinstance(packages, dict) or not packages
            or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                   for k, v in packages.items())):
        raise ValueError('Evidence lacks package version provenance')
    source_paths = sorted([*root.glob('*.py'), *(root / 'agfl').rglob('*.py'),
                           *(root / 'inm').rglob('*.py')])
    source_map = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source_paths}
    if not source_map or report.get('source_files_sha256') != source_map:
        raise ValueError('Study source changed or provenance is incomplete; use a fresh output')
    expected = {(subject, seed) for subject in range(1, 10) for seed in range(3)}
    if pilot:
        expected = {(1, 0)}
    entries = report.get('tasks')
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError(f'Evidence must contain exactly {len(expected)} task records')
    common = ('study_id', 'config_sha256', 'partitions', 'training_regime', 'source_files_sha256')
    seen, paths = set(), set()
    expected_rows = {(name, repeat) for name, count in CONDITIONS.items() for repeat in range(count)}
    for entry in entries:
        if not isinstance(entry, dict) or any(type(entry.get(k)) is not int for k in ('subject', 'seed')):
            raise ValueError('Malformed task identity')
        identity = (entry['subject'], entry['seed'])
        if identity not in expected or identity in seen:
            raise ValueError('Duplicate or unexpected evidence task')
        seen.add(identity)
        path = _checked_path(evidence_path.parent.parent, entry.get('path'), entry.get('sha256'), 'Task')
        if path in paths or path == evidence_path:
            raise ValueError('Task evidence paths must be unique')
        paths.add(path)
        task = read_json(path)
        if (not isinstance(task, dict) or task.get('schema_name') != 'agfl-encoder-candidates-task-v1'
                or any(type(task.get(k)) is not int for k in ('subject', 'seed'))
                or (task.get('subject'), task.get('seed')) != identity
                or task.get('status') != 'complete' or task.get('synthetic') is not False
                or any(task.get(k) != report[k] for k in common)
                or not _hash(task.get('split_id')) or not _hash(task.get('data_id'))):
            raise ValueError(f'Invalid task evidence identity: {path}')
        checks = task.get('checks')
        if not isinstance(checks, dict) or set(checks) != CHECKS or any(v is not True for v in checks.values()):
            raise ValueError('Unresolved task checks')
        fits = task.get('fits')
        if not isinstance(fits, list) or len(fits) != len(ARMS):
            raise ValueError('Each task requires exactly five fit records')
        seen_arms, fit_paths, paired_masks = set(), set(), None
        for fit_record in fits:
            if not isinstance(fit_record, dict):
                raise ValueError('Malformed fit record')
            arm = fit_record.get('arm')
            if arm not in ARMS or arm in seen_arms:
                raise ValueError('Duplicate or unexpected fit arm')
            seen_arms.add(arm)
            fit_path = _checked_path(path.parent, fit_record.get('path'), fit_record.get('sha256'), 'Fit')
            if fit_path in fit_paths or fit_path == path:
                raise ValueError('Fit paths must be unique')
            fit_paths.add(fit_path)
            fit = read_json(fit_path)
            fit_common = ('subject', 'seed', 'study_id', 'config_sha256', 'split_id', 'data_id', 'partitions')
            if (not isinstance(fit, dict) or fit.get('arm') != arm or fit.get('status') != 'complete'
                    or fit.get('synthetic') is not False
                    or any(type(fit.get(k)) is not int for k in ('subject', 'seed'))
                    or any(fit.get(k) != task[k] for k in fit_common)
                    or type(fit.get('selected_epoch')) is not int or not 1 <= fit['selected_epoch'] <= 250
                    or type(fit.get('parameter_count')) is not int or fit['parameter_count'] < 1):
                raise ValueError('Invalid fit identity, selection, or parameter count')
            _metrics(fit.get('clean_train'), 'clean training')
            artifacts = fit.get('artifact_sha256')
            required = {'checkpoint.pt', 'history.json', 'predictions.npz'}
            if arm.startswith('local_'):
                required |= {'probe.npz', 'probe_shuffled.npz'}
                if fit.get('probe_converged') is not True or fit.get('probe_shuffled_converged') is not True:
                    raise ValueError('Local probe diagnostics are incomplete')
            if not isinstance(artifacts, dict) or not required <= set(artifacts):
                raise ValueError('Required checkpoint/history/prediction/probe artifacts missing')
            artifact_paths = set()
            for relative, expected_hash in artifacts.items():
                artifact = _checked_path(fit_path.parent, relative, expected_hash, 'Fit artifact')
                if artifact in artifact_paths or artifact == fit_path:
                    raise ValueError('Artifact paths must be unique and not refer to the fit record')
                artifact_paths.add(artifact)
            rows = fit.get('validation')
            if not isinstance(rows, list) or len(rows) != len(expected_rows):
                raise ValueError('Incomplete validation condition coverage')
            masks = {}
            for row in rows:
                if (not isinstance(row, dict) or not isinstance(row.get('scenario'), str)
                        or type(row.get('repeat')) is not int):
                    raise ValueError('Malformed validation condition')
                key = (row['scenario'], row['repeat'])
                if key not in expected_rows or key in masks or not _hash(row.get('mask_sha256')):
                    raise ValueError('Invalid validation condition or mask identity')
                _metrics(row, 'validation')
                masks[key] = row['mask_sha256']
            if paired_masks is not None and masks != paired_masks:
                raise ValueError('Evaluation masks are not paired across arms')
            paired_masks = masks
    print(f'Session {sid}: real {len(expected)}-task/{5*len(expected)}-fit evidence and artifacts verified')


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
        discovered = loader.discover(str(root / "tests/encoder_candidates"), pattern=pattern)
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


def execution_gate(root, state, phase, device):
    """Require reviewed software readiness before an explicit scientific command."""
    import importlib.util
    for sid in ('01', '02'):
        check_receipt(root, state, sid)
    path = root / 'docs/candidate-followup/readiness.json'
    readiness = read_json(path)
    checks = {'rng_independence', 'repeat_fit_equivalence', 'all_five_smoke', 'pilot_input_verified'}
    if (not isinstance(readiness, dict) or readiness.get('status') != 'ready'
            or set(readiness.get('checks', {})) != checks
            or any(v is not True for v in readiness['checks'].values())
            or not isinstance(readiness.get('verified_devices'), list)
            or not readiness['verified_devices']
            or any(v not in ('cpu', 'cuda') for v in readiness['verified_devices'])
            or device not in readiness['verified_devices']):
        raise ValueError('Execution readiness or requested-device validation is incomplete')
    module_path = root / 'inm/encoder_candidates/protocol.py'
    spec = importlib.util.spec_from_file_location('followup_candidate_protocol', module_path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    # Execution continues the reviewed pilot/cohort output. Loading is read-only;
    # the identity and artifact checks below still govern whether it is reusable.
    cfg = protocol.load_config(root / 'configs/encoder-candidates-reproducible.json',
                               allow_existing_output=True)
    identity = protocol.study_identity(cfg)
    if (cfg.get('synthetic') is not False or cfg.get('subjects') != list(range(1, 10))
            or cfg.get('seeds') != [0, 1, 2]
            or Path(cfg['output_dir']).resolve() != root/'results/encoder-candidates-reproducible-v1'):
        raise ValueError('Execution requires the fixed real reproducible cohort config/output')
    for key in ('study_id', 'config_sha256', 'source_files_sha256', 'packages'):
        if readiness.get(key) != identity.get(key):
            raise ValueError(f'Readiness is stale: {key} changed; review before real execution')
    artifacts = readiness.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Readiness requires saved verification logs and hashes')
    for name, checksum in artifacts.items():
        _checked_path(root/'docs/candidate-followup', name, checksum, 'Readiness artifact')
    if phase == 'cohort':
        check_receipt(root, state, '03')
        gate(root, '03')
        review = read_json(root/'docs/candidate-followup/pilot-review.json')
        task_path = root/'results/encoder-candidates-reproducible-v1/tasks/A01_seed_0/task.json'
        if (review.get('decision') != 'proceed' or review.get('blockers') != []
                or not isinstance(review.get('summary'), str) or not review['summary'].strip()
                or review.get('study_id') != identity['study_id']
                or review.get('device') != device
                or review.get('pilot_sha256') != hashlib.sha256(task_path.read_bytes()).hexdigest()):
            raise ValueError('Cohort requires a current pilot review with decision proceed and no blockers')
        task = read_json(task_path)
        for entry in task['fits']:
            fit = read_json(task_path.parent / entry['path'])
            if fit.get('execution_device') != device:
                raise ValueError('Pilot and cohort must use the same verified execution device')
    print(f'{phase}: reviewed software readiness verified for {device}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "ids", "default", "prompt", "fingerprint",
                                         "validate-report", "verify", "receipt", "check-receipt", "gate", "execution-gate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--phase", choices=["pilot", "cohort", "summarize"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
    elif args.action == "execution-gate":
        if args.phase is None or args.state is None:
            raise ValueError("execution-gate requires --phase and --state")
        execution_gate(root, args.state.resolve(), args.phase, args.device)
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
