"""Fault injection in launcher control flow; no scientific fixtures or results."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

class LauncherSequence(unittest.TestCase):
    def test_failed_first_fit_audit_stops_next_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); stub=folder/'record_child'; events=folder/'events.jsonl'
            stub.write_text('#!/usr/bin/env python3\nimport sys,os,json\nargs=sys.argv[1:]\nwith open(os.environ["AUDIT_STUB_EVENTS"],"a") as f:f.write(json.dumps(args)+"\\n")\nraise SystemExit(37 if "audit-fit-task" in args else 0)\n')
            stub.chmod(0o700)
            env=os.environ.copy();env['PUBLISHED_COVARIANCE_PYTHON']=str(stub);env['AUDIT_STUB_EVENTS']=str(events)
            p=subprocess.run(['bash','scripts/run_published_covariance.sh','pilot','--output',str(folder/'uncreated'),'--device','cuda:0'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
            self.assertEqual(p.returncode,37,p.stdout+p.stderr)
            rows=[json.loads(x) for x in events.read_text().splitlines()]
            fits=[r for r in rows if 'fit-task' in r]
            audits=[r for r in rows if 'audit-fit-task' in r]
            self.assertEqual(len(fits),1,rows)
            self.assertEqual(len(audits),1,rows)
            self.assertEqual(fits[0][fits[0].index('--seed')+1],'0')
            self.assertEqual(audits[0][audits[0].index('--seed')+1],'0')
            self.assertFalse((folder/'uncreated').exists(),'Stub unexpectedly created scientific output')

if __name__=='__main__':unittest.main(verbosity=2)
