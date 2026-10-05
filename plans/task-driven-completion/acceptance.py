"""Supervisor-owned mandatory discovery gate; workers must not edit."""
import os, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
sid=sys.argv[1]
patterns={'01':'test_protocol.py','02':'test_models.py','03':'test_*.py','04':'test_*.py'}
suite=unittest.TestLoader().discover(str(ROOT/'tests/task_driven_completion'),pattern=patterns[sid])
count=suite.countTestCases()
if count==0: raise SystemExit('No tests discovered')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped: raise SystemExit(1)
print(f'Mandatory acceptance: {count} tests, no skips')
