"""Public synthetic lifecycle, immutable resume, fit-seal and corruption checks."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def digest_tree(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}

class ExecutionAcceptance(unittest.TestCase):
    def call(self,mode,output,*extra,ok=True):
        p=subprocess.run([sys.executable,'-m','published_covariance',mode,'--synthetic','--output',str(output),'--device','cpu',*extra],cwd=ROOT,capture_output=True,text=True,timeout=180)
        if ok:self.assertEqual(p.returncode,0,p.stdout+'\n'+p.stderr)
        else:self.assertNotEqual(p.returncode,0,p.stdout+'\n'+p.stderr)
        return p

    def test_complete_synthetic_lifecycle_and_faults(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-study'
            self.call('pilot',output)
            self.call('mark-pilot',output)
            before=digest_tree(output);self.assertTrue(before)
            self.call('evaluate-cohort',output,ok=False)
            self.assertEqual(digest_tree(output),before,'Premature E mode modified the study')
            self.call('review-pilot',output)
            self.call('train-cohort',output)
            outside=Path(tmp)/'external'; outside.mkdir()
            link=output/'evaluation'; link.symlink_to(outside,target_is_directory=True)
            self.call('evaluate-cohort',output,ok=False)
            self.assertEqual(list(outside.iterdir()),[],'Nested output symlink allowed an external write')
            link.unlink()
            self.call('evaluate-cohort',output)
            self.call('audit',output)
            self.call('summarize',output)
            before=digest_tree(output)
            self.call('pilot',output)
            self.call('mark-pilot',output)
            self.call('summarize',output)
            self.assertEqual(digest_tree(output),before,'Completed reuse or summary changed scientific bytes')
            cell=output/'evaluation/A01-s0-zero-static_16-r0.npz'
            original_cell=cell.read_bytes();cell.unlink()
            self.call('evaluate-task',output,'--subject','A01','--seed','0',ok=False)
            self.assertFalse(cell.exists(),'Partial evaluation silently regenerated a missing cell')
            task=output/'evaluation-tasks/A01-s0.json';original_task=task.read_bytes();task.unlink()
            self.call('evaluate-task',output,'--subject','A01','--seed','0',ok=False)
            self.assertFalse(cell.exists(),'Partial unreceipted evaluation was silently completed')
            cell.write_bytes(original_cell);task.write_bytes(original_task)
            # A valid zip with altered probabilities must fail numerical replay.
            cell=sorted((output/'evaluation').glob('*-zero-static_16-r0.npz'))[0]
            original=cell.read_bytes()
            with np.load(cell,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
            arrays['probabilities']=np.roll(arrays['probabilities'],1,axis=1)
            with cell.open('wb') as stream: np.savez_compressed(stream,**arrays)
            self.call('audit',output,ok=False)
            self.call('summarize',output,ok=False)
            cell.write_bytes(original)
            # A self-consistent empty row index must not certify an evaluated cohort.
            rowfile=output/'evaluation-rows.npz'; original=rowfile.read_bytes()
            auditfile=output/'audit.json'; audit_original=auditfile.read_bytes(); auditfile.unlink()
            with rowfile.open('wb') as stream: np.savez_compressed(stream,json_rows=np.asarray('[]'))
            self.call('audit',output,ok=False)
            rowfile.write_bytes(original)
            auditfile.write_bytes(audit_original)
            # The row index is an aggregation of audited cells, not a free source of metrics.
            original=rowfile.read_bytes();auditfile.unlink()
            with np.load(rowfile,allow_pickle=False) as z: index=json.loads(str(z['json_rows'].item()))
            index[0]['balanced_accuracy'] += .125
            with rowfile.open('wb') as stream: np.savez_compressed(stream,json_rows=np.asarray(json.dumps(index)))
            self.call('audit',output,ok=False)
            rowfile.write_bytes(original);auditfile.write_bytes(audit_original)
            human=output/'report.md'; original=human.read_bytes()
            human.write_text('Tampered conclusion\n')
            self.call('summarize',output,ok=False)
            human.write_bytes(original)
            # A corrupted selected numeric artifact must stop audit and resume.
            candidates=list((output/'fits').glob('*.npz'))
            self.assertTrue(candidates,'No persisted numerical state exists')
            artifact=sorted(candidates)[0]
            artifact.write_bytes(artifact.read_bytes()+b'corruption')
            self.call('audit',output,ok=False)
            self.call('pilot',output,ok=False)
            foreign=Path(tmp)/'foreign';foreign.mkdir();(foreign/'unrelated.txt').write_text('preserve me')
            self.call('pilot',foreign,ok=False)
            self.assertEqual((foreign/'unrelated.txt').read_text(),'preserve me')
            link=Path(tmp)/'symlink';link.symlink_to(foreign,target_is_directory=True)
            self.call('pilot',link,ok=False)

if __name__=='__main__':unittest.main(verbosity=2)
