"""Independent saved-fit admission; never invokes a classifier or changes outputs."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--pilot',action='store_true');a=p.parse_args()
r=a.output;sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
identity=json.loads((r/'identity.json').read_text());assert identity['synthetic_fixture'] is False
assert all(sha(Path(k))==v for k,v in identity['source'].items())
assert not (r/'evaluation').exists(),'Admission must precede E predictions'
seal=None
if not a.pilot:
 seal=json.loads((r/'fit-seal.json').read_text());assert seal['task_count']==27
 assert seal['identity_sha256']==sha(r/'identity.json')
rows=[]
for subject in (['A01'] if a.pilot else [f'A{i:02d}' for i in range(1,10)]):
 states=[]
 for seed in range(3):
  f=r/'fits'/f'{subject}-s{seed}.npz';m=json.loads(f.with_suffix('.json').read_text())
  assert m['subject']==subject and m['seed']==seed and m['synthetic_fixture'] is False
  assert m['classifier_state_sha256_before']==m['classifier_state_sha256_after'];states.append(m['classifier_state_sha256_before'])
  assert m['npz_sha256']==sha(f)
  audit=json.loads((r/'fit-audits'/f'{subject}-s{seed}.json').read_text())
  assert audit=={'schema':1,'identity_sha256':sha(r/'identity.json'),'subject':subject,'seed':seed,'fit_sha256':sha(f),'passed':True}
  if seal:
   bound=next(t for t in seal['tasks'] if t['subject']==subject and t['seed']==seed)
   assert bound['sha256']==sha(f) and bound['metadata_sha256']==sha(f.with_suffix('.json'))
  with np.load(f,allow_pickle=False) as z:
   history=json.loads(str(z['history']));assert len(history)==m['epochs_completed']+1
   assert [h['epoch'] for h in history]==list(range(len(history)))
   chosen=min(history,key=lambda h:(-h['degraded_ba'],h['degraded_log_loss'],h['epoch']))
   assert chosen['epoch']==m['selected_epoch']==int(z['selected_epoch'])
   assert np.isfinite(z['selected_free']).all() and z['selected_free'].shape==(253,)
   assert len(z['train_ids'])==232 and len(z['validation_ids'])==56
   assert set(z['train_ids']).isdisjoint(z['validation_ids'])
   assert all(':T:' in str(i) for i in np.concatenate((z['train_ids'],z['validation_ids'])))
   rng=np.random.Generator(np.random.PCG64(seed));valid=[]
   for c in range(4):
    ix=np.flatnonzero(z['all_labels']==c);rng.shuffle(ix);valid.extend(ix[:14])
   np.testing.assert_array_equal(z['validation_indices'],sorted(valid))
   assert all(np.isfinite(h['gradient_norm_mean']) and h['gradient_norm_mean']>0 for h in history[1:])
   assert 10<=m['epochs_completed']<=100
  rows.append({'subject':subject,'seed':seed,'epochs_completed':m['epochs_completed'],'selected_epoch':m['selected_epoch'],'fit_sha256':sha(f),'classifier_unchanged':True})
 assert len(set(states))==1
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'pilot':a.pilot,'fit_count':len(rows),'identity_sha256':sha(r/'identity.json'),'fit_seal_sha256':sha(r/'fit-seal.json') if seal else None,'E_predictions_before_admission':0,'epoch_zero_selections':sum(t['selected_epoch']==0 for t in rows),'epoch_budget_saturations':sum(t['epochs_completed']==100 for t in rows),'fits':rows},indent=2))
