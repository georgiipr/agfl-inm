"""Dependency-free supervisor check of the declared public task plan."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
code='''
import sys,runpy
class NoScience:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','sklearn','tensorflow','torch','mne','h5py'}:
            raise RuntimeError('Plan imported scientific dependency: '+fullname)
sys.meta_path.insert(0,NoScience())
sys.path.insert(0,sys.argv[1])
sys.argv=['published_covariance','plan']
runpy.run_module('published_covariance',run_name='__main__')
'''
result=subprocess.run([sys.executable,'-I','-S','-c',code,str(root)],cwd=root,capture_output=True,text=True,timeout=20)
assert result.returncode==0,(result.stdout,result.stderr)
plan=json.loads(result.stdout)
assert plan['covariance_fits']==27 and plan['backbone_fits']==0 and plan['arm_evaluations']==81,plan
assert len(plan['tasks'])==27
assert {(r['subject'],r['seed']) for r in plan['tasks']}=={(f'A{i:02}',s) for i in range(1,10) for s in range(3)}
print('PASS dependency-light plan: 27 covariance fits, 0 backbone fits, 81 arm evaluations')
