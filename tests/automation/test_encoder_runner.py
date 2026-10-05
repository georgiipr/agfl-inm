"""Check the encoder workflow in isolated workspaces without real model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import test_accuracy_runner as fixtures


class EncoderRunnerTests(fixtures.AccuracyRunnerTests):
    """Reuse the established failure/resume suite against the independent runner."""

    def setUp(self):
        super().setUp()
        for name in ("run_encoder_sessions.sh", "encoder_session_state.py"):
            shutil.copy2(fixtures.ROOT / "scripts" / name, self.root / "scripts" / name)
        previous = self.plan
        self.plan = self.root / "plans/encoder-audit"
        shutil.copytree(previous, self.plan)
        for name in ("COMMON.md", "CONTRACT.md", "result.schema.json"):
            shutil.copy2(fixtures.ROOT / "plans/encoder-audit" / name, self.plan / name)
        fake = fixtures.FAKE_CODEX.replace("Execute accuracy improvement session",
                                            "Execute encoder investigation session")
        fake = fake.replace("tests/baselines", "tests/encoder_audit")
        fake = fake.replace("plans/accuracy/CONTRACT.md", "plans/encoder-audit/CONTRACT.md")
        self.fake.write_text(fake)
        self.script = self.root / "scripts/run_encoder_sessions.sh"

    def completed(self, sid):
        return (self.root / f".session-runs/encoder-audit/completed/{sid}.json").is_file()

    def add_evidence_gate(self):
        manifest_path = self.plan / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sessions"][0]["evidence"] = "results/encoder-audit-v1/report/evidence.json"
        manifest_path.write_text(json.dumps(manifest))

    def make_evidence(self):
        self.add_evidence_gate()
        output = self.root / "results/encoder-audit-v1"
        (output / "report").mkdir(parents=True, exist_ok=True)
        report = {"schema_name": "agfl-encoder-audit-report-v1", "status": "complete",
                  "synthetic": False, "partitions": ["train", "validation"],
                  "subjects": list(range(1, 10)), "seeds": [0, 1, 2],
                  "input_study_ids": {"baseline": "a" * 64, "legacy": "b" * 64},
                  "source_files_sha256": {"run.py": hashlib.sha256(
                      (self.root / "run.py").read_bytes()).hexdigest()}, "tasks": []}
        for subject in range(1, 10):
            for seed in range(3):
                task = {"subject": subject, "seed": seed, "status": "complete",
                        "synthetic": False, "partitions": report["partitions"],
                        "input_study_ids": report["input_study_ids"],
                        "checks": {name: True for name in (
                            "source_integrity", "trial_alignment", "split_isolation",
                            "normalization_replay", "checkpoint_replay", "probe_controls", "mask_pairing")}}
                for section in ("alignment", "clean_checkpoints", "probes", "reconstruction"):
                    task[section] = {"fixture": "test evidence, never a real measurement"}
                relative = f"tasks/A{subject:02d}_seed_{seed}/audit.json"
                path = output / relative
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(task))
                report["tasks"].append({"subject": subject, "seed": seed, "path": relative,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        path = output / "report/evidence.json"
        path.write_text(json.dumps(report))
        return path, report

    def run_gate(self):
        return subprocess.run([sys.executable, str(self.root / "scripts/encoder_session_state.py"),
                               "gate", "--root", str(self.root), "--session", "01"],
                              text=True, capture_output=True, timeout=5)

    def test_exact_session_shorthand(self):
        result = self.run_script("--session", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c["id"] for c in self.calls()], ["01"])
        self.assertTrue(self.completed("01"))
        self.assertFalse(self.completed("02"))

    def test_completion_requires_checks_and_distinguishes_readiness_findings(self):
        report = {"session_id": "01", "status": "completed",
                  "summary": "Inventory software implemented and verified.",
                  "files_changed": [], "checks": ["Synthetic inventory tests passed."],
                  "blockers": [], "next_session_notes":
                  "Real replay remains blocked by historical source mismatches."}
        path = self.root / "handoff.json"

        def validate(value):
            path.write_text(json.dumps(value))
            return subprocess.run(
                [sys.executable, str(self.root / "scripts/encoder_session_state.py"),
                 "validate-report", "--root", str(self.root), "--session", "01",
                 "--report", str(path)], text=True, capture_output=True, timeout=5)

        valid = validate(report)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        for changes, message in (
                ({"blockers": ["Historical source mismatch"]}, "blockers is nonempty"),
                ({"summary": "  "}, "nonempty summary"),
                ({"checks": []}, "actual checks"),
                ({"checks": ["  "]}, "actual checks"),
                ({"status": "blocked", "blockers": ["Acceptance failed"]}, "did not complete")):
            with self.subTest(changes=changes):
                result = validate(dict(report, **changes))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_session_and_range_options_cannot_be_combined(self):
        for options in (("--session", "01", "--through", "02"),
                        ("--from", "01", "--session", "02")):
            result = self.run_script(*options)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not both", result.stderr)
        self.assertFalse(self.calls())

    def test_missing_real_evidence_stops_before_model_call(self):
        self.add_evidence_gate()
        result = self.run_script("--session", "01")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evidence.json", result.stderr)
        self.assertFalse(self.calls())
        self.assertFalse(self.completed("01"))

    def test_complete_evidence_and_invalid_cohort_metadata(self):
        path, report = self.make_evidence()
        valid = self.run_gate()
        self.assertEqual(valid.returncode, 0, valid.stderr)
        for field, value in (("synthetic", True), ("synthetic", 0),
                             ("status", "partial"), ("subjects", [1]),
                             ("partitions", ["train", "validation", "test"]),
                             ("input_study_ids", {}), ("source_files_sha256", {}),
                             ("tasks", report["tasks"][:-1]),
                             ("tasks", [report["tasks"][0]] * 27)):
            with self.subTest(field=field, value=str(value)[:40]):
                invalid = dict(report, **{field: value})
                path.write_text(json.dumps(invalid))
                self.assertNotEqual(self.run_gate().returncode, 0)

        report["subjects"][0] = True
        path.write_text(json.dumps(report))
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_evidence_task_tampering_and_failed_checks(self):
        path, report = self.make_evidence()
        record = report["tasks"][0]
        task_path = path.parent.parent / record["path"]
        task = json.loads(task_path.read_text())
        task["checks"]["normalization_replay"] = False
        task_path.write_text(json.dumps(task))
        tampered = self.run_gate()
        self.assertNotEqual(tampered.returncode, 0)
        self.assertIn("checksum mismatch", tampered.stderr)
        record["sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(report))
        failed = self.run_gate()
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("diagnostic checks", failed.stderr)

    def test_evidence_cannot_reference_outside_output(self):
        path, report = self.make_evidence()
        report["tasks"][0]["path"] = "../../run.py"
        path.write_text(json.dumps(report))
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside the audit output", result.stderr)

    def test_current_source_mismatch_blocks_evidence_review(self):
        self.make_evidence()
        with (self.root / "run.py").open("a") as stream:
            stream.write("\n# changed research code\n")
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source changed", result.stderr)

    def test_future_plan_edit_also_invalidates_receipts(self):
        result = self.run_script("--session", "01")
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.plan / "02.md").open("a") as stream:
            stream.write("\nnew downstream requirement\n")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale", result.stderr)

    def test_baseline_regression_is_independent_and_cannot_skip(self):
        path = self.plan / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["sessions"][0]["baseline_regression"] = True
        path.write_text(json.dumps(manifest))
        tests = self.root / "tests/baselines"
        tests.mkdir(parents=True)
        (tests / "test_baseline.py").write_text(
            "import unittest\nclass Baseline(unittest.TestCase):\n"
            "    def test_baseline(self): self.skipTest('must fail acceptance')\n")
        result = self.run_script("--session", "01")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.completed("01"))


if __name__ == "__main__":
    import unittest
    unittest.main()
