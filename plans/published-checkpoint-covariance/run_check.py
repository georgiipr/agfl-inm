"""Supervisor command receipts, immutable logs and bounded process groups."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--receipt',required=True,type=Path)
p.add_argument('--timeout',type=float,default=600)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
assert command and a.timeout>0
prefix=a.receipt
record={'command':command,'cwd':os.getcwd(),'start_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'timeout_seconds':a.timeout}
start=time.monotonic()
with Path(str(prefix)+'.stdout').open('xb') as out,Path(str(prefix)+'.stderr').open('xb') as err:
    child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
    record['pid']=child.pid
    try:
        code=child.wait(timeout=a.timeout)
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
        record['termination']=type(exc).__name__
        os.killpg(child.pid,signal.SIGTERM)
        time.sleep(20)
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait();code=124 if isinstance(exc,subprocess.TimeoutExpired) else 130
record.update(exit=code,end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
with Path(str(prefix)+'.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
raise SystemExit(code)
