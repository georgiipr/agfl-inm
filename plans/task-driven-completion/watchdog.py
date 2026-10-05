"""Supervisor-only inventories, deadlines and review receipts."""
import sys,json,hashlib,subprocess,datetime,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; STATE=ROOT/'.session-runs/task-driven-completion'
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def inventory():
 paths=set(ROOT/x for x in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
 for name in ('agfl','inm','configs','docs','plans','scripts','tests','skills','results','.session-runs'):
  paths.update((ROOT/name).rglob('*'))
 return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts and not p.relative_to(ROOT).as_posix().startswith('.session-runs/task-driven-completion/')}
def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def allowed(sid,p):
 doc=f'docs/task-driven-completion/session-{sid}-handoff.md'
 scopes={
 '01':['inm/task_driven_completion/__init__.py','inm/task_driven_completion/protocol.py','inm/task_driven_completion/__main__.py','configs/task-driven-completion.json','tests/task_driven_completion/test_protocol.py','docs/task-driven-completion/protocol.md',doc],
 '02':['inm/task_driven_completion/completion.py','inm/task_driven_completion/adapters.py','tests/task_driven_completion/test_models.py',doc],
 '03':['inm/task_driven_completion/training.py','inm/task_driven_completion/study.py','inm/task_driven_completion/reporting.py','inm/task_driven_completion/__main__.py','tests/task_driven_completion/test_training.py','tests/task_driven_completion/test_study.py',doc],
 '04':['inm/task_driven_completion/','tests/task_driven_completion/','docs/task-driven-completion/']}
 return any(p.startswith(x) if x.endswith('/') else p==x for x in scopes[sid])
cmd,sid=sys.argv[1:3]; folder=STATE/sid
if cmd=='start':
 folder.mkdir(exist_ok=False)
 now=time.time(); before=inventory(); write(folder/'before.json',before)
 write(folder/'start.json',{'session':sid,'start_unix':now,'deadline_unix':now+1200,'start_utc':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),'worker_model':'inherited current model','plan_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'plans/task-driven-completion').rglob('*') if p.is_file()}})
 print(json.dumps({'session':sid,'deadline_utc':datetime.datetime.fromtimestamp(now+1200,datetime.timezone.utc).isoformat(),'files':len(before)}))
elif cmd=='status':
 row=json.loads((folder/'start.json').read_text()); now=time.time()
 print(json.dumps({'session':sid,'elapsed_seconds':round(now-row['start_unix']),'remaining_seconds':round(row['deadline_unix']-now)}))
elif cmd=='review':
 protected=json.loads((STATE/'protected.json').read_text()); before=json.loads((folder/'before.json').read_text()); after=inventory()
 changed=sorted(p for p in set(before)|set(after) if before.get(p)!=after.get(p))
 violations=[p for p in changed if not allowed(sid,p)]
 historical=[p for p,h in protected.items() if after.get(p)!=h]
 row={'session':sid,'changed':changed,'out_of_scope':violations,'historical_changes':historical,'protected_files':len(protected),'protection_passed':not violations and not historical}
 write(folder/'protection-review.json',row); print(json.dumps(row,indent=2))
 if violations or historical: raise SystemExit(1)
