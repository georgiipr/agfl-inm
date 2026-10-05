"""Verify historical and CPU/CUDA full-input replay for all tasks, without fits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import copy
import json
import tempfile
import numpy as np
import torch
from inm.task_driven_completion_cuda.protocol import load_config, tasks
from inm.task_driven_completion_cuda.execution import scoped_cuda_rng, execution_metadata
from inm.completion_transformer.replay import prepare_task, restore_backbones, replay_full_input
from inm.task_driven_completion.adapters import CompletionAdapter
from inm.task_driven_completion.training import predict as cpu_predict
from inm.task_driven_completion_cuda.training import predict as cuda_predict

with scoped_cuda_rng(920001), tempfile.TemporaryDirectory(prefix='completion-prefit-all-') as tmp:
    cfg=load_config('configs/task-driven-completion-cuda-v2.json', allow_existing_output=True)
    cfg['output_dir']=tmp
    print(json.dumps({'execution':execution_metadata(),'new_fits':0}),flush=True)
    for spec in tasks(cfg):
        subject,seed=spec['subject'],spec['seed']
        prepared,record=prepare_task(cfg,subject,seed,Path(tmp)/f'A{subject:02d}_{seed}')
        replay=replay_full_input(record,prepared)
        model=restore_backbones(record)['spatial_transformer']
        x=torch.from_numpy(np.array(prepared.validation.raw,dtype=np.float32,copy=True))
        mask=np.ones((len(x),22,4),dtype=np.bool_)
        cpu=cpu_predict(CompletionAdapter(model),x,mask)
        gpu=cuda_predict(CompletionAdapter(copy.deepcopy(model).cuda()),x,mask)
        passed=bool(np.allclose(cpu,gpu,rtol=1e-4,atol=1e-6))
        print(json.dumps({'subject':subject,'seed':seed,'historical_replay':replay,
            'max_abs_probability_error':float(np.abs(cpu-gpu).max()),'passed':passed}),flush=True)
        if not passed: raise RuntimeError('CPU/CUDA prefit replay failed')
