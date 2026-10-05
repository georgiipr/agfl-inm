# Candidate training reproducibility

Session 01 reproduced and repaired uncontrolled training-time randomness in
the separate encoder-candidate package. A one-epoch `spatial_eegnet` fit with
seed 0 and identical constructor weights produced different histories after
ambient Torch seeds 123 and 456 before the repair. The regression now compares
actual fit histories, selected CPU state tensors, and validation probabilities
from separate output directories.

## Seed policy

The training policy is `agfl-candidate-fit-rng-v1` (version 1). For task seed
`s`, Python, NumPy's global generator, the CPU Torch generator, and (when used)
the selected CUDA generator receive
`(s + 1,900,001) mod (2^63 - 1)`. Python and NumPy global states are saved and
restored; Torch uses `torch.random.fork_rng` for CPU and only the selected CUDA
device. Restoration also occurs if fitting raises an exception.

Minibatch order has an independent NumPy `default_rng` stream for each epoch:
`(s + 100,003 * epoch + 700,001) mod (2^63 - 1)`, with one-based epoch numbers.
It does not depend on parameter count, dropout draws, arm position, or other
fits. Candidate constructors now seed only Torch's CPU default generator
inside their existing CPU RNG fork. This prevents `torch.manual_seed()` from
also touching unrelated CUDA generators during CPU-only initialization. The
five model architectures and their tensors/settings are unchanged.

The fit scope enables deterministic Torch algorithms, sets cuDNN deterministic
mode, and disables cuDNN benchmarking. It restores the caller's prior settings
on both success and failure. CUDA runs require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before the first CUDA context. If CUDA was
already initialized without that setting, the fit fails clearly instead of
silently weakening determinism. The setting is temporarily established before
CUDA initialization when possible, then the caller's environment is restored.

Each fit result and full checkpoint record the policy/version, task/effective
seeds, data-order formula, Python/NumPy/PyTorch versions, deterministic flags,
and `execution_device` (`cpu` or `cuda`). This metadata is part of reuse
compatibility, so an artifact with a different RNG policy or device cannot be
reused as a matching fit. Numeric checkpoint state remains loadable with
`weights_only=True`.

## Verification and limits

CPU synthetic checks cover the original dropout failure, actual short fits of
all five models under different ambient RNG histories, fit independence after
another arm, different task seeds, a fresh subprocess, caller RNG/backend-state
restoration after success and exception, and checkpoint policy round-trip and
incompatible-reuse rejection. Tests compare histories, selected states, and
probabilities, excluding timing and filesystem paths. Existing candidate test
coverage is run alongside these checks.

CUDA was unavailable in the session environment (`torch.cuda.is_available()`
was false), so no CUDA numerical repeatability test was run and CUDA is not
verified. There is no promise of bitwise agreement across device types, device
models, PyTorch/package versions, or hardware. The demonstrated guarantee is
repeatability on the same device and software environment; deterministic
operations that are unsupported by a requested CUDA stack will fail rather
than be relaxed.
