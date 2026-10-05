"""Supervisor-owned sequential real CUDA task/audit watchdog; no retries."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STATE=ROOT/'.session-runs/task-driven-completion-cuda/experiments-v2'
CONFIG=ROOT/'configs/task-driven-completion-cuda-v2.json'

def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def hash_file(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def emit(event,**fields):
 row={'time':now(),'event':event,**fields};print(json.dumps(row),flush=True)
 with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
def protection():
 recorded=json.loads((ROOT/'.session-runs/task-driven-completion-cuda/repair-01/protected.json').read_text())
 bad=[p for p,h in recorded.items() if not (ROOT/p).is_file() or hash_file(ROOT/p)!=h]
 if bad:raise RuntimeError('Protected historical files changed: '+repr(bad))
 return len(recorded)
def identity():
 sys.path.insert(0,str(ROOT))
 from inm.task_driven_completion_cuda.protocol import load_config,study_identity
 cfg=load_config(CONFIG,allow_existing_output=True)
 actual=study_identity(cfg)
 expected=json.loads((ROOT/'docs/task-driven-completion-cuda/frozen-real-identity-v2.json').read_text())
 if actual!=expected:raise RuntimeError('Frozen CUDA source/config/package/device identity differs')
 return cfg,actual

def resources(child_pid):
 row={'time':now(),'child_pid':child_pid}
 try:
  fields=Path(f'/proc/{child_pid}/status').read_text().splitlines()
  row['process_memory']=[s for s in fields if s.startswith(('VmRSS:','VmHWM:'))]
  result=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
  row['gpu']=result.stdout.strip();row['nvidia_smi_exit_code']=result.returncode
 except (OSError,subprocess.TimeoutExpired) as exc:row['collection_error']=repr(exc)
 with (STATE/'resources.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

def command(args,stem,timeout):
 stdout=STATE/(stem+'.stdout.log');stderr=STATE/(stem+'.stderr.log')
 if stdout.exists() or stderr.exists():raise RuntimeError('Attempt logs exist; refusing overwrite/retry: '+stem)
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 started=time.time(); emit('command_start',name=stem,command=args,timeout_seconds=timeout)
 with stdout.open('x') as out,stderr.open('x') as err:
  child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
  timed_out=False
  try:
   while child.poll() is None:
    if time.time()-started>=timeout:
     timed_out=True;os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     break
    atomic(STATE/'status.json',{'status':'running','phase':stem,'pid':os.getpid(),'child_pid':child.pid,'elapsed_seconds':time.time()-started,'started_utc':now()})
    resources(child.pid)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:emit('heartbeat',name=stem,elapsed_seconds=round(time.time()-started))
  except BaseException:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
 receipt={'name':stem,'command':args,'exit_code':child.returncode,'timed_out':timed_out,'elapsed_seconds':time.time()-started,'stdout':str(stdout),'stderr':str(stderr),'finished_utc':now()}
 atomic(STATE/(stem+'.command.json'),receipt)
 emit('command_finish',**receipt)
 if timed_out or child.returncode!=0:raise RuntimeError('Command failed; cohort stopped: '+stem)
 return receipt

def review(cfg,task_index,fit_receipt,audit_receipt):
 subject=task_index//3+1;seed=task_index%3
 path=Path(cfg['output_dir'])/'tasks'/f'A{subject:02d}_seed_{seed}'/'task.json'
 task=json.loads(path.read_text());audit=json.loads(Path(audit_receipt['stdout']).read_text())
 if task.get('status')!='complete' or task.get('synthetic') is not False:raise RuntimeError('Not complete real evidence')
 if task.get('subject')!=subject or task.get('seed')!=seed:raise RuntimeError('Task index identity mismatch')
 if task.get('replay',{}).get('cuda_full_input_vs_cpu',{}).get('passed') is not True:raise RuntimeError('Task CPU/CUDA full-input check failed')
 if audit.get('cuda_full_input_vs_cpu',{}).get('passed') is not True:raise RuntimeError('Audit CPU/CUDA full-input check failed')
 if len(task['cells'])!=105 or len(task['fits'])!=5:raise RuntimeError('Incomplete strategy/cell grid')
 if audit.get('status')!='passed' or audit.get('synthetic') is not False or audit.get('replayed_cells')!=105 or audit.get('new_fits')!=0:raise RuntimeError('Independent audit incomplete')
 if audit.get('device')!='cuda:0' or task.get('execution',{}).get('device')!='cuda:0':raise RuntimeError('GPU execution evidence missing')
 if audit.get('study_id')!=task.get('study_id'):raise RuntimeError('Audit study identity mismatch')
 for key in ('backbone_matches_source','selected_states_loaded','exact_prediction_replay','independently_recomputed_metrics','independently_recomputed_contrasts'):
  if audit.get(key) is not True:raise RuntimeError('Required audit check missing: '+key)
 count=protection()
 record={'task_index':task_index,'subject':subject,'seed':seed,'status':'accepted','decision':'proceed','study_id':task['study_id'],'task_sha256':hash_file(path),'audit_sha256':hash_file(Path(audit_receipt['stdout'])),'fit_command':fit_receipt,'audit_command':audit_receipt,'protected_files_unchanged':count,'fits':task['fits'],'reviewed_utc':now(),'review_basis':'operational correctness and independent prediction audit; accuracy is not an advancement gate'}
 atomic(STATE/f'task-{task_index:02d}.review.json',record);return record

def main():
 def interrupted(signum, frame): raise KeyboardInterrupt(f'watchdog signal {signum}')
 signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pilot','cohort']);parser.add_argument('--detach',action='store_true');parser.add_argument('--task-timeout',type=int,default=7200);args=parser.parse_args()
 STATE.mkdir(parents=True,exist_ok=True)
 if args.detach:
  log=STATE/(args.phase+'-watchdog.log')
  if log.exists():raise RuntimeError('Watchdog attempt already exists; no implicit retry')
  with log.open('x') as stream:
   child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),args.phase,'--task-timeout',str(args.task_timeout)],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps({'status':'started','phase':args.phase,'watchdog_pid':child.pid,'log':str(log)}));return
 import fcntl
 with (STATE/'watchdog.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   cfg,frozen=identity();protection()
   if args.phase=='cohort':
    gate=json.loads((STATE/'pilot-review.json').read_text())
    if gate.get('decision')!='proceed_to_cohort' or gate.get('study_id')!=frozen['study_id']:raise RuntimeError('Independent pilot review gate not passed')
   indices=[0] if args.phase=='pilot' else range(1,27)
   for i in indices:
    identity()
    if (STATE/f'task-{i:02d}.review.json').exists():raise RuntimeError('Task already has review; no implicit restart')
    fit=command([sys.executable,'-m','inm.task_driven_completion_cuda','--config',str(CONFIG),'--task-index',str(i)],f'task-{i:02d}-fit',args.task_timeout)
    audit=command([sys.executable,'-m','inm.task_driven_completion_cuda.audit','--config',str(CONFIG),'--task-index',str(i)],f'task-{i:02d}-audit',1200)
    receipt=review(cfg,i,fit,audit);emit('task_accepted',task_index=i,study_id=receipt['study_id'])
   if args.phase=='cohort':
    command([sys.executable,'-m','inm.task_driven_completion_cuda','--config',str(CONFIG),'--summarize-only'],'cohort-report',1200)
   atomic(STATE/'status.json',{'status':'pilot_awaiting_review' if args.phase=='pilot' else 'cohort_complete','phase':args.phase,'pid':os.getpid(),'finished_utc':now(),'study_id':frozen['study_id']})
   emit('phase_complete',phase=args.phase)
  except BaseException as exc:
   atomic(STATE/'status.json',{'status':'stopped','phase':args.phase,'pid':os.getpid(),'finished_utc':now(),'error':repr(exc)})
   emit('stopped',phase=args.phase,error=repr(exc));raise
if __name__=='__main__':main()
