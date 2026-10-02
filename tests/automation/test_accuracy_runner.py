"""Exercise orchestration without an account, network, model calls, or EEG packages."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]

FAKE_CODEX = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import sys
import time

args = sys.argv[1:]
root = Path(args[args.index('--cd') + 1])
output = Path(args[args.index('--output-last-message') + 1])
prompt = sys.stdin.read()
sid = re.search(r'Execute accuracy improvement session (\d+):', prompt).group(1)
with (root / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'id': sid, 'args': args, 'prompt': prompt,
                             'python': os.environ.get('AGFL_PYTHON')}) + '\n')
mode = os.environ.get('AGFL_FAKE_MODE', 'success')
if sid != os.environ.get('AGFL_FAKE_ID', '01'):
    mode = 'success'
if mode == 'exit':
    print('deliberate CLI failure', file=sys.stderr)
    sys.exit(7)
if mode == 'timeout':
    time.sleep(30)
if mode == 'invalid':
    output.write_text('not json')
    sys.exit(0)
if mode != 'no_artifact':
    (root / f'artifact-{sid}.txt').write_text('implemented')
tests = root / 'tests/baselines'
tests.mkdir(parents=True, exist_ok=True)
body = 'self.assertTrue(True)'
if mode == 'test_fail':
    body = 'self.fail("deliberate acceptance failure")'
if mode == 'skip':
    body = 'self.skipTest("not evidence of completion")'
(tests / f'test_{sid}.py').write_text(
    'import unittest\nclass Acceptance(unittest.TestCase):\n'
    '    def test_result(self):\n        ' + body + '\n')
if mode == 'no_tests':
    (tests / f'test_{sid}.py').write_text('# no tests\n')
if mode == 'plan_edit':
    with (root / 'plans/accuracy/CONTRACT.md').open('a') as stream:
        stream.write('\nchanged by model\n')
report = {'session_id': '99' if mode == 'wrong_id' else sid,
          'status': 'blocked' if mode == 'blocked' else 'completed',
          'summary': f'Implemented fixture {sid}',
          'files_changed': [f'artifact-{sid}.txt'],
          'checks': ['fixture implementation checked'],
          'blockers': ['missing dependency'] if mode == 'blocked' else [],
          'next_session_notes': f'interface_from_{sid}'}
output.write_text(json.dumps(report))
print(json.dumps({'type': 'turn.completed'}))
'''


class AccuracyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agfl runner ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        for name in ("run_accuracy_sessions.sh", "accuracy_session_state.py"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.plan = self.root / "plans/accuracy"
        self.plan.mkdir(parents=True)
        for name in ("COMMON.md", "CONTRACT.md", "result.schema.json"):
            shutil.copy2(ROOT / "plans/accuracy" / name, self.plan / name)
        sessions = []
        for sid in ("01", "02"):
            (self.plan / f"{sid}.md").write_text(f"Fixture task {sid}\n")
            sessions.append({"id": sid, "title": f"Fixture {sid}", "prompt": f"{sid}.md",
                             "artifacts": [f"artifact-{sid}.txt"], "tests": [f"test_{sid}.py"]})
        (self.plan / "manifest.json").write_text(json.dumps(
            {"version": 1, "default_through": "02", "sessions": sessions}))
        (self.root / "run.py").write_text(
            'for i in range(27):\n'
            '    print(f"{i}: A{i//3+1:02d}, seed={i%3}, classifier fits=14")\n'
            'print("Total classifier fits: 378")\n')
        (self.root / ".gitignore").write_text("/.session-runs/\n__pycache__/\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        self.fake = self.root / "fake codex"
        self.fake.write_text(FAKE_CODEX)
        self.fake.chmod(0o755)
        self.env = dict(os.environ, AGFL_PYTHON=sys.executable, AGFL_CODEX_BIN=str(self.fake))
        self.script = self.root / "scripts/run_accuracy_sessions.sh"

    def run_script(self, *arguments, **env):
        return subprocess.run(["bash", str(self.script), *arguments], cwd="/tmp",
                              env=dict(self.env, **env), text=True, capture_output=True, timeout=20)

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def completed(self, sid):
        return (self.root / f".session-runs/accuracy/completed/{sid}.json").is_file()

    def test_list_and_dry_run_do_not_call_or_write_state(self):
        for args in [("--list",), ("--dry-run",), ("--help",)]:
            result = self.run_script(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls())
        self.assertFalse((self.root / ".session-runs").exists())

    def test_sequential_handoff_and_resume(self):
        first = self.run_script("--through", "1")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(self.completed("01"))
        self.assertFalse(self.completed("02"))
        second = self.run_script("--from", "02", "--model", "test-model")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual([c["id"] for c in self.calls()], ["01", "02"])
        self.assertIn("interface_from_01", self.calls()[1]["prompt"])
        self.assertIn("test-model", self.calls()[1]["args"])
        third = self.run_script()
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_project_environment_default_and_explicit_override(self):
        interpreter = self.root / '.venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        default = self.run_script('--through', '01', AGFL_PYTHON='')
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(self.calls()[-1]['python'], str(interpreter))
        override = self.run_script('--from', '02', '--python', sys.executable,
                                   AGFL_PYTHON='')
        self.assertEqual(override.returncode, 0, override.stderr)
        self.assertEqual(self.calls()[-1]['python'], sys.executable)

    def test_failure_stops_before_next_session(self):
        for mode in ("exit", "invalid", "blocked", "wrong_id", "no_artifact", "no_tests", "test_fail", "skip"):
            with self.subTest(mode=mode):
                state = f".session-runs/{mode}"
                result = self.run_script("--state-dir", state, AGFL_FAKE_MODE=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls()[-1]["id"], "01")
                self.assertFalse((self.root / state / "completed/01.json").exists())
                # Avoid an artifact from an earlier attempt satisfying the missing-artifact case.
                for path in self.root.glob("artifact-*.txt"):
                    path.unlink()
        self.assertTrue(all(c["id"] == "01" for c in self.calls()))

    def test_timeout_and_retry_with_different_model(self):
        failed = self.run_script("--timeout", "1", AGFL_FAKE_MODE="timeout")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("124", failed.stderr)
        self.assertFalse(self.completed("01"))
        resumed = self.run_script("--model", "replacement", "--through", "01")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertTrue(self.completed("01"))
        self.assertEqual([c["id"] for c in self.calls()], ["01", "01"])

    def test_predecessors_cannot_be_skipped(self):
        result = self.run_script("--from", "02")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prerequisites", result.stderr)
        self.assertFalse(self.calls())

    def test_stale_receipt_and_in_session_plan_edit_stop(self):
        result = self.run_script("--through", "01")
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.plan / "01.md").open("a") as stream:
            stream.write("new requirement\n")
        stale = self.run_script()
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("stale", stale.stderr)
        changed = self.run_script("--state-dir", ".session-runs/edited", AGFL_FAKE_MODE="plan_edit")
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("modified its plan", changed.stderr)

    def test_model_argument_is_not_shell_code(self):
        marker = self.root / "should-not-exist"
        malicious = f"model; touch '{marker}'"
        result = self.run_script("--through", "01", "--model", malicious)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(malicious, self.calls()[0]["args"])
        self.assertFalse(marker.exists())

    def test_invalid_arguments(self):
        for args in [("--model",), ("--from", "99"), ("--from", "02", "--through", "01"),
                     ("--timeout", "0"), ("--effort", "anything"), ("--unknown",)]:
            with self.subTest(args=args):
                self.assertNotEqual(self.run_script(*args).returncode, 0)
        self.assertFalse(self.calls())

    def test_workspace_lock_applies_across_state_directories(self):
        lock = self.root / ".session-runs/accuracy.lock"
        lock.parent.mkdir()
        import fcntl
        with lock.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script("--state-dir", ".session-runs/another")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another accuracy-session runner", result.stderr)
        self.assertFalse(self.calls())


if __name__ == "__main__":
    unittest.main()
