"""Scoped deterministic CUDA execution; never silently fall back to CPU."""
from contextlib import contextmanager
import os
import random

import numpy as np
import torch

DEVICE = 'cuda:0'
CUBLAS_CONFIG = ':4096:8'
_initialized_safely = False


@contextmanager
def scoped_cuda_rng(seed, *, threads=1):
    global _initialized_safely
    if type(seed) is not int or seed < 0 or type(threads) is not int or threads < 1:
        raise ValueError('seed must be nonnegative and threads positive integers')
    previous_env = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    # A caller that initialized CUDA with another setting cannot establish that
    # cuBLAS handles were created deterministically; fail rather than guess.
    if torch.cuda.is_initialized() and previous_env != CUBLAS_CONFIG and not _initialized_safely:
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    thread_count = torch.get_num_threads()
    flags = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.mha.get_fastpath_enabled())
    gpu_state = None
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS_CONFIG
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA cuda:0 is required; CPU fallback is forbidden')
        gpu_state = torch.cuda.get_rng_state(0)
        _initialized_safely = True
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        # Do not use manual_seed_all or torch.manual_seed: other GPUs are unrelated.
        torch.cuda.default_generators[0].manual_seed(seed)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Fused Transformer inference fails the fixed historical CPU/CUDA replay
        # tolerance. Use the same unfused path throughout this execution scope.
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.cuda.device(0), torch.autocast('cuda', enabled=False):
            yield torch.device(DEVICE)
    finally:
        if gpu_state is not None:
            torch.cuda.set_rng_state(gpu_state, 0)
        torch.random.set_rng_state(cpu_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(thread_count)
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[4:6]
        torch.backends.mha.set_fastpath_enabled(flags[6])
        if previous_env is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = previous_env


def execution_metadata():
    """Call only within the deterministic scope, on the actual selected GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA cuda:0 is required')
    mha_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or mha_fastpath_enabled
            or torch.is_autocast_enabled('cuda') or torch.get_num_threads() != 1
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != CUBLAS_CONFIG):
        raise RuntimeError('CUDA execution settings differ from the fixed deterministic protocol')
    props = torch.cuda.get_device_properties(0)
    return dict(device=DEVICE, calibration_device='cpu', historical_replay_device='cpu',
        threads=1, gpu_name=props.name, gpu_capability=list(torch.cuda.get_device_capability(0)),
        gpu_total_memory=props.total_memory, gpu_uuid=str(getattr(props, 'uuid', 'unavailable')),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_benchmark=False, cudnn_deterministic=True, tf32=False, amp=False,
        mha_fastpath_enabled=mha_fastpath_enabled,
        cublas_workspace_config=CUBLAS_CONFIG,
        cross_device_tolerance={'rtol': 1e-4, 'atol': 1e-6})


def require_cuda_module(module):
    tensors = list(module.parameters()) + list(module.buffers())
    if any(t.device != torch.device(DEVICE) for t in tensors):
        raise ValueError('All execution model tensors must be on cuda:0')
