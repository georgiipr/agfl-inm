# CUDA task-driven completion execution

This is the separately authorized CUDA successor of the accepted CPU study.
The fixed declaration is `configs/task-driven-completion-cuda.json`; execution
lives in `inm/task_driven_completion_cuda/`. Older source, configurations, CPU
readiness receipts and scientific artifacts remain historical evidence unchanged.

The five strategies, 27 participant/seed tasks, 54 supervised completion fits,
30-epoch training-only Tucker initialization, ranks, losses, masks, optimizers,
selection and reporting are unchanged. This study still reuses validation for
historical checkpoint selection and completion selection; its results are not
independent confirmation. Preserve negative effects and incomplete coverage.

## Execution identity and gates

Use the project virtualenv. All numerical execution requires **cuda:0** and fails
if CUDA is unavailable. There is no CPU fallback. Calibration and strict original
historical prediction replay stay on CPU with one thread. Both historical
backbones replay before calibration; only the historical spatial Transformer is
used by the five completion strategies.

The runner also compares current CPU full-input probabilities against CUDA before
calibration or supervised fitting with fixed `rtol=1e-4, atol=1e-6`. Within CUDA,
full-input probabilities across all strategies and selected-state reloads must
be bitwise equal. Audit independently recomputes the CPU reference, checks the
cross-device tolerance, and reconstructs all 105 CUDA prediction cells.

CUDA scopes set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization,
enable deterministic algorithms with errors, disable cuDNN benchmarking and TF32,
and disable autocast. They restore Python, NumPy, CPU and cuda:0 RNG, thread count,
backend flags, current device, autocast and the environment on success or error.
They do not manually seed unrelated GPUs. If an external caller has initialized
CUDA before configuring cuBLAS, the scope refuses execution. Public commands own
their first CUDA initialization; embedding applications should set the same
cuBLAS environment before their own CUDA initialization.

Float64 covariance parameters and observed-entry ridge solves remain float64;
only device transfer is applied. Do not cast whole adapters to float32.
The actual GPU name, UUID where available, capability, memory, Torch/CUDA/cuDNN,
execution settings and source SHA are stored in task metadata. Resume, audit and
reporting require the same hardware/runtime. The study identity includes config,
packages, all reused/new Python source and historical manifest hashes. A source,
package, config or hardware change requires a fresh output identity.

Default real output: `results/task-driven-completion-cuda-v1`. Never reuse CPU
outputs. Completed tasks can resume only after identity, numerical artifact,
checksum and historical input verification. Partial/failed tasks cannot retry in
place. Retain failed attempts and their logs.

## Planning and synthetic acceptance

```bash
.venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --plan
.venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --preflight
.venv/bin/python -m unittest discover -s tests/task_driven_completion_cuda -p test_protocol.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/task_driven_completion_cuda -p test_gpu.py -v
```

Plan and preflight use only standard-library dependencies. They never claim
numerical readiness. The GPU suite must run where the actual GPU is accessible;
missing CUDA is an error, never a skip. It compares CPU/CUDA outputs and gradients,
hidden NaN/Inf and observed-sample identity, frozen-backbone classification
gradients, full bypass, all scoped state, and repeated tiny fits after changing
ambient RNG and arm order. It also runs public smoke/resume/audit in separate
processes and tests changed-identity refusal. All of this uses synthetic inputs.

A fresh smoke directory must be explicit. Substitute a fresh path for SMOKE:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --smoke --output-dir /tmp/SMOKE
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --smoke --output-dir /tmp/SMOKE --resume-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion_cuda.audit --config configs/task-driven-completion-cuda.json --synthetic-smoke --output-dir /tmp/SMOKE
```

Smoke creates four synthetic train and four validation trials, keeps CPU
calibration at 30 epochs, and explicitly limits supervised fitting to one epoch.
It writes five strategies and 105 prediction cells. Synthetic evidence is excluded
from real cohort reports. GPU acceptance and preservation must be recorded by the
supervisor before real execution; code availability is not readiness evidence.

## Real pilot, review and cohort

The user has already authorized the real experiment sequence. Real fitting is a
separate supervisor operation, outside the bounded coding session. Freeze the
current real identity after acceptance and record timed process commands, logs,
exit codes and hardware/resource use. Task 0 is A01 seed 0:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --task-index 0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion_cuda.audit --config configs/task-driven-completion-cuda.json --task-index 0
```

Review operational evidence: strict original replay, CPU/CUDA prefit agreement,
finite calibration/training histories, paired IDs and masks, unchanged backbone,
selected-state reload, exact independent CUDA audit, timing and resource use.
Pilot review is not an accuracy gate; do not tune settings after observing it.
A recorded operational pass permits task indexes 1 through 26, one timed process
at a time with its own independent audit and receipt. Stop on failure or timeout;
retain every attempt and do not automatically retry.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --summarize-only
```

Reporting averages repeats, then seeds within participant, then participants
with equal weights. It retains partial rows and negative effects, excludes
synthetic evidence and withholds real cohort means until all 27 tasks verify.
Incomplete reports return exit 2. The primary exploratory screen remains
learned Tucker minus frozen Tucker, at least 2 percentage points and 6/9 positive
participants, reported separately from comparisons with stronger controls.

## Recorded implementation acceptance

The supervisor ran the required GPU suite outside the sandbox on the NVIDIA
GeForce RTX 3060 Laptop GPU: **5 test groups passed, no skips, 23.009 seconds**,
first attempt. CPU structural checks passed 3/3 and compilation passed. The
unaltered log and checksum are retained under `validation/`. No source/config
changes followed the GPU acceptance. This establishes the checked synthetic
CUDA behavior; the supervisor still owns preservation verification, the frozen
real identity, readiness receipt and operational real pilot review.
