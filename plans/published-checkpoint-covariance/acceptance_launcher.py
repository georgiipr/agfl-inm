"""Exercise the actual Bash lock and process-group timeout implementation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/run_published_covariance.sh'

class LauncherAcceptance(unittest.TestCase):
    def test_occupied_shared_lock(self):
        with (ROOT/'.session-runs/accuracy.lock').open('rb') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=subprocess.run(['bash',str(SCRIPT),'smoke','--device','cpu'],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(p.returncode,0)
        self.assertTrue('lock' in (p.stdout+p.stderr).lower() or 'another' in (p.stdout+p.stderr).lower(),p.stdout+p.stderr)

    def test_timeout_cleans_term_ignoring_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);childfile=folder/'child.pid';parent=folder/'parent.py'
            parent.write_text("import subprocess,sys,time\np=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)'])\nopen(sys.argv[1],'w').write(str(p.pid))\ntime.sleep(90)\n")
            # Required helper is also the helper used for public launcher children.
            command='source "$1"; run_bounded "$2" 1 "$3" "$4" "$5"'
            p=subprocess.run(['bash','-c',command,'acceptance',str(SCRIPT),str(folder/'timeout'),sys.executable,str(parent),str(childfile)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertIn(p.returncode,[124,137],p.stdout+p.stderr)
            self.assertTrue(childfile.exists())
            pid=int(childfile.read_text())
            for _ in range(20):
                try:
                    state=Path(f'/proc/{pid}/stat').read_text().split()[2]
                    if state=='Z':break
                except FileNotFoundError:break
                time.sleep(.1)
            else:self.fail(f'timeout left live descendant PID {pid}')

if __name__=='__main__':unittest.main(verbosity=2)
