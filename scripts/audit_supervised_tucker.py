"""Independent replay/aggregation audit. Read-only except explicit /tmp JSON output."""
import argparse,hashlib,importlib.metadata,importlib.util,json,math,platform,sys,tempfile
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=='scripts' else Path.cwd()).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from inm.encoder_candidates.data import prepare_subject
from inm.supervised_tucker.models import restore_model,build_model
from inm.supervised_tucker.study import load_config
from inm.availability import make_mask_bank,mask_bank_digest,CHANNEL_IDS
spec=importlib.util.spec_from_file_location('oldaudit',ROOT/'scripts/audit_tensor_temporal_screen.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,sha,digest,metrics,predict=helper.read,helper.sha,helper.digest,helper.metrics,helper.predict

def statehash(state,keys=None):
 return digest({name:{'shape':list(state[name].shape),'dtype':str(state[name].dtype),'sha256':hashlib.sha256(state[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()} for name in sorted(state if keys is None else keys)})
def close(actual,expected):
 for key,value in actual.items():
  assert math.isclose(value,expected[key],abs_tol=1e-7,rel_tol=0),(key,value,expected[key])

def audit(cfg,indices=None):
 torch.set_num_threads(1)
 output=Path(cfg['output_dir']); manifest=read(output/'study.json'); identity=manifest['identity']
 assert digest(identity)==manifest['study_id']
 assert identity['config']=={k:v for k,v in cfg.items() if k!='config_path'}
 assert identity['config_sha256']==sha(cfg['config_path'])
 assert identity['python']==platform.python_version()
 for package,version in identity['packages'].items(): assert importlib.metadata.version(package)==version
 sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(ROOT.glob('*.py'))|set((ROOT/'agfl').rglob('*.py'))|set((ROOT/'inm').rglob('*.py')))}
 assert sources==identity['source_sha256']
 for name,value in identity['input_sha256'].items(): assert sha(name)==value
 tasks=[(s,z) for s in cfg['subjects'] for z in cfg['seeds']]
 if indices is not None:
  assert indices and len(indices)==len(set(indices)) and all(0<=i<len(tasks) for i in indices)
  tasks=[tasks[i] for i in indices]
 report={'study_id':manifest['study_id'],'source_files':len(sources),'tasks':[],'pending':[],'classifier_training_run':False,'training_only_initial_calibration_recomputed':True,'partitions_replayed':['train','validation'],'robustness_evaluations':0}
 rows=[];robustness=[]
 for subject,seed in tasks:
  td=output/'tasks'/f'A{subject:02d}_seed_{seed}'
  if not (td/'complete.json').exists(): report['pending'].append([subject,seed]);continue
  print(f'Independent audit A{subject:02d} seed {seed}',flush=True)
  completion=read(td/'complete.json'); provenance=read(td/'data_provenance.json')
  assert completion['study_id']==manifest['study_id'] and completion['subject']==subject and completion['seed']==seed
  assert completion['data_sha256']==sha(td/'data_provenance.json')
  assert set(completion['results_sha256'])==set(cfg['arms'])
  with tempfile.TemporaryDirectory(prefix='agfl-supervised-audit-',dir='/tmp') as tmp: prepared=prepare_subject(cfg,subject,seed,tmp)
  assert provenance['data_id']==prepared.data_id and provenance['split_id']==prepared.split_id
  assert provenance['normalization']==prepared.normalization
  for partition in ('train','validation'):
   data=getattr(prepared,partition)
   assert provenance['partitions'][partition]['sample_ids']==list(data.sample_ids)
   for field,arr in [('raw',data.raw),('labels',data.labels)]:assert provenance['partitions'][partition][field]==hashlib.sha256(arr.tobytes()).hexdigest()
  initial=build_model(cfg['arms'][0],seed)
  recalibration=initial.calibrate(torch.tensor(np.asarray(prepared.train.raw),dtype=torch.float32),tucker_epochs=cfg['calibration_epochs'])
  initial_hash=statehash(initial.state_dict()); initial_factors_hash=statehash(initial.state_dict(),('U','V'))
  initial_calibration_hash=digest({k:v for k,v in recalibration.items() if k not in ('arm_id','supervised_factors')})
  paired=[];pairmasks=None
  for arm in cfg['arms']:
   ad=td/arm; result=read(ad/'result.json'); history=read(ad/'history.json')
   assert completion['results_sha256'][arm]==sha(ad/'result.json')
   assert result['identity']=={'study_id':manifest['study_id'],'arm':arm,'synthetic':False,'data':provenance}
   for file,hashed in result['artifact_sha256'].items():assert sha(ad/file)==hashed
   assert [row['epoch'] for row in history]==list(range(1,len(history)+1))
   selected=max(history,key=lambda row:(row['clean_validation_metrics']['balanced_accuracy'],-row['clean_validation_metrics']['log_loss'],-row['epoch']))
   assert result['selected_epoch']==selected['epoch']
   assert [r['epoch'] for r in history if r['selected']]==[selected['epoch']]
   assert result['epochs_run']==len(history)
   payload=torch.load(ad/'checkpoint.pt',map_location='cpu',weights_only=True)
   assert payload['identity']==result['identity'] and payload['selected_epoch']==result['selected_epoch']
   assert payload['normalization']==prepared.normalization
   for key in ('initial_state_sha256','initial_factor_sha256','calibration_sha256'):assert payload[key]==result[key]
   assert result['initial_state_sha256']==initial_hash
   assert result['initial_factor_sha256']==initial_factors_hash
   assert result['calibration_sha256']==initial_calibration_hash
   cal=result['calibration']
   assert digest({k:v for k,v in cal.items() if k not in ('arm_id','supervised_factors')})==result['calibration_sha256']
   assert len(cal['factor_history'])==cal['factor_epochs']==cfg['calibration_epochs']
   assert cal['training_trials']==len(prepared.train.labels)
   factorhash=statehash(payload['state_dict'],('U','V'))
   assert factorhash==result['selected_factor_sha256']
   changed=factorhash!=result['initial_factor_sha256']
   assert changed==result['factors_changed']
   assert changed==(arm=='spectral_tucker_supervised')
   for key in ('U','V'): assert np.allclose(payload['state_dict'][key].norm(dim=0).numpy(),1,atol=2e-7,rtol=0)
   model=restore_model(arm,payload['constructor'],payload['state_dict']).eval()
   assert result['parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
   replay={}
   with np.load(ad/'predictions.npz',allow_pickle=False) as saved:
    for partition in ('train','validation'):
     data=getattr(prepared,partition); probability=predict(model,data.raw,cfg['training']['batch_size'])
     assert np.array_equal(saved[partition+'_labels'],data.labels)
     assert np.array_equal(saved[partition+'_sample_ids'],data.sample_ids)
     assert np.array_equal(saved[partition+'_probabilities'],probability)
     measured=metrics(data.labels,probability);close(measured,result[partition+'_metrics']);close(measured,selected['clean_'+partition+'_metrics'])
     replay[partition]=measured
   robust=read(ad/'robustness.json');masks=[(r['scenario'],r['repeat'],r['mask_sha256']) for r in robust]
   assert len(masks)==sum(c['repeats'] for c in cfg['conditions']) and len(set((s,r) for s,r,_ in masks))==len(masks)
   if pairmasks is not None:assert masks==pairmasks
   pairmasks=masks
   for condition in cfg['conditions']:
    values=[r for r in robust if r['scenario']==condition['name']]
    assert sorted(r['repeat'] for r in values)==list(range(condition['repeats']))
    for row in values:
     bank=make_mask_bank(len(prepared.validation.raw),4,condition['retained'],condition['pattern'],seed=seed,partition='validation',subject=f'A{subject:02d}',repeat=row['repeat'],sample_ids=prepared.validation.sample_ids,channel_ids=CHANNEL_IDS)
     assert mask_bank_digest(bank)==row['mask_sha256']
     prob=predict(model,prepared.validation.raw,cfg['training']['batch_size'],masks=bank)
     close(metrics(prepared.validation.labels,prob),row)
     report['robustness_evaluations']+=1
    robustness.append({'subject':subject,'seed':seed,'arm':arm,'scenario':condition['name'],**{k:float(np.mean([r[k] for r in values])) for k in ('balanced_accuracy','log_loss')}})
   rows.append({'subject':subject,'seed':seed,'arm':arm,**replay})
   paired.append(result)
  for field in ('initial_state_sha256','initial_factor_sha256','calibration_sha256'):assert paired[0][field]==paired[1][field]
  report['tasks'].append({'subject':subject,'seed':seed,'paired_initialization_exact':True,'frozen_factors_unchanged':True,'learned_factors_changed':True,'checkpoint_replay_bitwise':True})
 report['cohort']=[];report['contrasts']=[];report['robustness_cohort']=[]
 if len(report['tasks'])==len(cfg['subjects'])*len(cfg['seeds']):
  saved=read(output/'report/summary.json');assert saved['status']=='complete' and saved['study_id']==manifest['study_id']
  mean=lambda arm,subject,part,key:float(np.mean([r[part][key] for r in rows if r['arm']==arm and r['subject']==subject]))
  candidate,reference=cfg['comparisons']['primary']
  for arm in cfg['arms']:
   cohort={'arm':arm,'participants':len(cfg['subjects']),'seeds_per_participant':len(cfg['seeds'])}
   for partition,key in [('train','balanced_accuracy'),('validation','balanced_accuracy'),('validation','accuracy'),('validation','log_loss')]:cohort[partition+'_'+key]=float(np.mean([mean(arm,s,partition,key) for s in cfg['subjects']]))
   actual=next(r for r in saved['cohort'] if r['arm']==arm)
   close({k:v for k,v in cohort.items() if k!='arm'},actual);report['cohort'].append(cohort)
  for key in ('balanced_accuracy','log_loss'):
   delta=np.array([mean(candidate,s,'validation',key)-mean(reference,s,'validation',key) for s in cfg['subjects']])
   rng=np.random.default_rng(cfg['comparisons']['bootstrap_seed'])
   boot=np.mean(delta[rng.integers(len(delta),size=(cfg['comparisons']['bootstrap_repeats'],len(delta)))],axis=1)
   result={'metric':key,'mean_difference':float(delta.mean()),'ci95_participant_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'participant_differences':dict(zip(map(str,cfg['subjects']),delta.tolist()))}
   actual=next(r for r in saved['contrasts'] if r['metric']==key)
   assert math.isclose(result['mean_difference'],actual['mean_difference'],abs_tol=1e-7,rel_tol=0)
   assert np.allclose(result['ci95_participant_bootstrap'],actual['ci95_participant_bootstrap'],atol=1e-7,rtol=0)
   if key=='balanced_accuracy':
    result['positive_participants']=int((delta>0).sum());assert result['positive_participants']==actual['positive_participants']
    result['passes_prespecified_screen']=bool(delta.mean()>=cfg['comparisons']['screening_mean_gain'] and (delta>0).sum()>=cfg['comparisons']['screening_positive_participants']);assert result['passes_prespecified_screen']==actual['passes_prespecified_screen']
   report['contrasts'].append(result)
  report['robustness_contrasts']=[]
  for scenario in [c['name'] for c in cfg['conditions']]:
   for key in ('balanced_accuracy','log_loss'):
    delta=np.array([np.mean([r[key] for r in robustness if r['subject']==subject and r['arm']==candidate and r['scenario']==scenario])-np.mean([r[key] for r in robustness if r['subject']==subject and r['arm']==reference and r['scenario']==scenario]) for subject in cfg['subjects']])
    rng=np.random.default_rng(cfg['comparisons']['bootstrap_seed'])
    boot=delta[rng.integers(len(delta),size=(cfg['comparisons']['bootstrap_repeats'],len(delta)))].mean(1)
    row={'scenario':scenario,'metric':key,'mean_difference':float(delta.mean()),'ci95_participant_bootstrap':np.quantile(boot,[.025,.975]).tolist()}
    actual=next(r for r in saved['robustness_contrasts'] if r['scenario']==scenario and r['metric']==key)
    assert math.isclose(row['mean_difference'],actual['mean_difference'],abs_tol=1e-7,rel_tol=0)
    assert np.allclose(row['ci95_participant_bootstrap'],actual['ci95_participant_bootstrap'],atol=1e-7,rtol=0)
    report['robustness_contrasts'].append(row)
  for arm in cfg['arms']:
   for scenario in [c['name'] for c in cfg['conditions']]:
    row={'arm':arm,'scenario':scenario,**{key:float(np.mean([np.mean([r[key] for r in robustness if r['subject']==s and r['arm']==arm and r['scenario']==scenario]) for s in cfg['subjects']])) for key in ('balanced_accuracy','log_loss')}}
    actual=next(r for r in saved['robustness_cohort'] if r['arm']==arm and r['scenario']==scenario);close({k:row[k] for k in ('balanced_accuracy','log_loss')},actual)
    report['robustness_cohort'].append(row)
 assert sources=={name:sha(ROOT/name) for name in sources}
 report['status']='passed' if not report['pending'] else 'partial'
 return report
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/supervised-tucker.json'));parser.add_argument('--task-indices',type=int,nargs='+');parser.add_argument('--output',default='/tmp/agfl-supervised-tucker-output-audit.json');args=parser.parse_args()
 assert Path(args.output).resolve().is_relative_to('/tmp')
 result=audit(load_config(args.config),args.task_indices)
 Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('status','robustness_evaluations','cohort','contrasts')},indent=2))
