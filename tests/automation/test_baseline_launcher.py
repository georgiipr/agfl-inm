"""Check interpreter/device dispatch without running experiments or needing a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FAKE_PYTHON = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['AGFL_LAUNCH_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), 'exe': sys.argv[0]}) + '\\n')
if args[0] == '-c':
    print(str(Path(sys.argv[0]).absolute()))
elif args[0].endswith('check_cuda.py'):
    sys.exit(int(os.environ.get('AGFL_PROBE_EXIT', '0')))
'''


class BaselineLauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='agfl launch ')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'scripts').mkdir()
        self.script = self.root / 'scripts/run_baselines.sh'
        shutil.copy2(ROOT / 'scripts/run_baselines.sh', self.script)
        self.interpreter = self.root / '.venv/bin/python'
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text(FAKE_PYTHON)
        self.interpreter.chmod(0o755)
        self.log = self.root / 'calls.jsonl'

    def run_launcher(self, *args, **env):
        return subprocess.run(['bash', str(self.script), *args], cwd='/tmp',
            env=dict(os.environ, AGFL_PYTHON='', AGFL_LAUNCH_LOG=str(self.log)) | env,
            text=True, capture_output=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_real_task_uses_project_environment_and_checks_cuda(self):
        result = self.run_launcher('--task-index', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[1]['args'][0].endswith('/check_cuda.py'))
        self.assertEqual(calls[2]['args'], ['-m', 'inm.baselines', '--task-index', '0'])
        self.assertEqual(calls[2]['cwd'], str(self.root))
        self.assertEqual(calls[2]['exe'], str(self.interpreter))

    def test_failed_cuda_probe_never_launches_training(self):
        result = self.run_launcher('--task-index=0', AGFL_PROBE_EXIT='7')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.calls()), 2)

    def test_cpu_and_nontraining_operations_do_not_probe_cuda(self):
        for args in [('--task-index', '0', '--device', 'cpu'),
                     ('--task-index=0', '--device=cpu'), ('--smoke',),
                     ('--plan',), ('--preflight',), ('--summarize-only',), ('--help',)]:
            with self.subTest(args=args):
                result = self.run_launcher(*args, AGFL_PROBE_EXIT='7')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.calls()[-1]['args'], ['-m', 'inm.baselines', *args])
        self.assertFalse(any(c['args'][0].endswith('check_cuda.py') for c in self.calls()))

    def test_explicit_interpreter_and_literal_arguments(self):
        alternate = self.root / 'custom python'
        shutil.copy2(self.interpreter, alternate)
        config = 'config with spaces; $(touch never-created).json'
        result = self.run_launcher('--config', config, '--plan', AGFL_PYTHON=str(alternate))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1]['exe'], str(alternate))
        self.assertEqual(self.calls()[-1]['args'], ['-m', 'inm.baselines', '--config', config, '--plan'])

    def test_standalone_probe(self):
        result = self.run_launcher('--check-cuda')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue(self.calls()[-1]['args'][0].endswith('/check_cuda.py'))
        self.assertEqual(self.run_launcher('--check-cuda', '--plan').returncode, 2)


if __name__ == '__main__':
    unittest.main()
