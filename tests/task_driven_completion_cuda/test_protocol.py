"""Dependency-light declaration, source identity and path isolation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inm.task_driven_completion import protocol as cpu
from inm.task_driven_completion_cuda import protocol as cuda

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/task-driven-completion-cuda-v2.json'


class ProtocolChecks(unittest.TestCase):
    def test_science_unchanged_and_execution_explicit(self):
        old,new=copy.deepcopy(cpu.FIXED['protocol']),copy.deepcopy(cuda.FIXED['protocol'])
        old.pop('execution'); execution=new.pop('execution')
        self.assertEqual(old,new)
        self.assertEqual(execution['device'],'cuda:0')
        self.assertEqual(execution['calibration_device'],'cpu')
        self.assertFalse(execution['amp']); self.assertFalse(execution['tf32'])
        self.assertIs(execution['mha_fastpath_enabled'], False)
        self.assertNotEqual(cpu.SCHEMA,cuda.SCHEMA)
        previous = json.loads((ROOT/'configs/task-driven-completion-cuda.json').read_text())
        revised = json.loads(CONFIG.read_text())
        self.assertEqual(revised['name'], 'task-driven-completion-cuda-v2')
        self.assertEqual(revised['output_dir'], '../results/task-driven-completion-cuda-v2')
        previous['protocol']['execution']['mha_fastpath_enabled'] = False
        previous.update(name=revised['name'], output_dir=revised['output_dir'])
        self.assertEqual(previous, revised)
        cfg=cuda.load_config(CONFIG,allow_existing_output=True)
        plan=cuda.plan(cfg)
        self.assertEqual((plan['tasks'],plan['supervised_completion_fits']),(27,54))
        self.assertFalse(plan['real_fitting_ready'])

    def test_import_plan_without_numerical_dependencies(self):
        code = '''import sys
class Guard:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','numpy','scipy','sklearn','mne'}:
   raise RuntimeError('numerical import forbidden: '+fullname)
sys.meta_path.insert(0,Guard())
from inm.task_driven_completion_cuda.protocol import load_config, plan, study_identity
cfg=load_config('configs/task-driven-completion-cuda-v2.json',allow_existing_output=True)
assert plan(cfg)['tasks']==27
assert 'inm/task_driven_completion/completion.py' in study_identity(cfg)['source_files_sha256']
assert 'inm/task_driven_completion_cuda/execution.py' in study_identity(cfg)['source_files_sha256']
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_cpu_outputs_and_science_changes_refused(self):
        cfg=cuda.load_config(CONFIG,allow_existing_output=True)
        cfg['output_dir']=str(ROOT/'results/task-driven-completion-v1')
        # Protect the old namespace even if its directory is not created yet.
        with self.assertRaises(ValueError):
            cuda.validate_output_path(cfg)
        # Old execution declarations must not be silently reused under this source.
        with self.assertRaisesRegex(ValueError, 'mha_fastpath_enabled'):
            cuda.load_config(ROOT/'configs/task-driven-completion-cuda.json', allow_existing_output=True)
        with tempfile.TemporaryDirectory() as directory:
            for field, key, value in (('training', 'learning_rate', .002),
                                      ('execution', 'mha_fastpath_enabled', True),
                                      ('execution', 'mha_fastpath_enabled', 0)):
                with self.subTest(field=field, key=key, value=value):
                    changed=json.loads(CONFIG.read_text())
                    changed['protocol'][field][key]=value
                    path=Path(directory)/'bad.json'; path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, key):
                        cuda.load_config(path)


if __name__ == '__main__':
    unittest.main()
