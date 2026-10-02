#!/usr/bin/env python3
"""Verify CUDA in this process with a small forward/backward GPU computation."""
from __future__ import annotations

import sys


def main():
    print(f"Python: {sys.executable}", flush=True)
    try:
        import torch
    except ImportError as error:
        print(f"PyTorch import failed: {error}. Use the project's prepared .venv.", file=sys.stderr)
        return 1
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}", flush=True)
    if torch.version.cuda is None:
        print("This interpreter has CPU-only PyTorch. Install a CUDA-enabled build in this environment.",
              file=sys.stderr)
        return 1
    try:
        # init() gives the driver error instead of collapsing all failures into False.
        torch.cuda.init()
        device = torch.cuda.current_device()
        x = torch.ones((32, 32), device=f"cuda:{device}", requires_grad=True)
        (x @ x).mean().backward()
        torch.cuda.synchronize(device)
        if x.grad is None or not torch.isfinite(x.grad).all().item():
            raise RuntimeError("GPU backward produced missing or nonfinite gradients")
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU {device}: {properties.name}; VRAM: {properties.total_memory / 2**30:.1f} GiB")
        print("CUDA forward/backward check passed.")
        return 0
    except (RuntimeError, AssertionError) as error:
        print(f"CUDA check failed in this process: {error}\n"
              "If the host GPU works, rerun from a terminal with GPU access or request "
              "outside-sandbox execution. Also check the NVIDIA driver and CUDA_VISIBLE_DEVICES. "
              "This does not establish that the host lacks CUDA.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
