"""Read-only checkpoint numerical diagnosis; no fitting or outcome selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import copy
import json
import tempfile
import numpy as np
import torch
from inm.task_driven_completion_cuda.protocol import load_config
from inm.task_driven_completion_cuda.execution import scoped_cuda_rng
from inm.completion_transformer.replay import prepare_task, restore_backbones, replay_full_input
from inm.task_driven_completion.adapters import CompletionAdapter

def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(t).softmax(-1).cpu() for t in x.split(32)]).numpy()

def difference(a,b):
    d=np.abs(a-b)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                failed_elements=int((d>1e-6+1e-4*np.abs(b)).sum()),
                argmax_changes=int((a.argmax(1)!=b.argmax(1)).sum()))

with scoped_cuda_rng(920001), tempfile.TemporaryDirectory(prefix='completion-diagnostic-') as tmp:
    cfg=load_config('configs/task-driven-completion-cuda.json', allow_existing_output=True)
    cfg['output_dir']=tmp
    prepared,record=prepare_task(cfg,1,0,Path(tmp)/'prepared')
    replay=replay_full_input(record,prepared)
    model=restore_backbones(record)['spatial_transformer']
    x=torch.from_numpy(np.array(prepared.validation.raw,dtype=np.float32,copy=True))
    cpu=predict(CompletionAdapter(copy.deepcopy(model)),x)
    gpu_model=CompletionAdapter(copy.deepcopy(model).cuda())
    gpu=predict(gpu_model,x.cuda())
    result={'historical_replay':replay,'default':difference(cpu,gpu)}
    torch.backends.mha.set_fastpath_enabled(False)
    cpu_slow=predict(CompletionAdapter(copy.deepcopy(model)),x)
    gpu_slow=predict(gpu_model,x.cuda())
    result['nonfused_cpu_gpu']=difference(cpu_slow,gpu_slow)
    result['cpu_fused_nonfused']=difference(cpu,cpu_slow)
    result['gpu_fused_nonfused']=difference(gpu,gpu_slow)
    torch.backends.cudnn.enabled=False
    gpu_native=predict(gpu_model,x.cuda())
    result['gpu_no_cudnn_vs_cpu']=difference(cpu,gpu_native)
    print(json.dumps(result,indent=2))
