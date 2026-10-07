# Explicit CUDA compiler discovery: recovery of the incomplete run

**Completed:** the fresh study passed all gates; see the [scientific verdict](verdict.md).
The output names below now exist. For a new reproduction, replace every occurrence
of `results/published-covariance-v1-cuda-root` with a new unused
`results/published-covariance-*` name. Never delete or overwrite either historical
output. Running compatible commands on completed outputs audits/reuses them; it
does not constitute a fresh refit.

The first attempt, `results/published-covariance-v1`, remains immutable and
incomplete. It contains 27 audited T-only covariance fits and 13 audited E tasks.
A05 seed 1 aborted with exit 134 before writing any task cells, because
TensorFlow tried to open `libdevice.10.bc` through a shared-library file as if it
were a directory. No complete report or scientific verdict was produced.
See [the bounded diagnosis](runtime-failure-handoff.md).

The repair uses the already installed CUDA 12.2 compiler wheel. Set both discovery
paths before every process that imports TensorFlow:

```bash
export PUBLISHED_CUDA_ROOT=/home/kalexu97/Projects/AGFL/.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$PUBLISHED_CUDA_ROOT"
export PATH="$PUBLISHED_CUDA_ROOT/bin:$PATH"
```

This preserves packages and scientific source/configuration. `XLA_FLAGS` changes
the canonical runtime identity, so use the fresh output
`results/published-covariance-v1-cuda-root`. Do not copy fits, predictions, audits,
or seals from the incomplete output. Refit all 27 tasks under the unchanged
scientific declaration, with a new three-seed pilot review before the remainder.
The prior partial E exposure remains disclosed; it is not erased by a new path.
No E outcomes were inspected to choose this compiler-discovery repair.

Independent generated acceptance consists of two fresh explicitly compiled
float64 math/gradient probes and all-nine native forward/backward plus a short
synthetic covariance fit/save/replay. Receipts and compiler asset hashes are in
`.session-runs/published-checkpoint-covariance/runtime-repair/`. The exact `ptxas`
resolution and PATH are recorded there in addition to canonical runtime identity.

After accepted runtime readiness, use the public launcher in this order, with
the environment above inherited by every command:

```bash
bash scripts/run_published_covariance.sh preflight --device cuda:0
bash scripts/run_published_covariance.sh reference --output results/published-covariance-v1-cuda-root --device cuda:0
bash scripts/run_published_covariance.sh pilot --output results/published-covariance-v1-cuda-root --device cuda:0
bash scripts/run_published_covariance.sh review-pilot --output results/published-covariance-v1-cuda-root --device cuda:0
# Supervisor reviews pilot correctness/resources before the remaining fits.
bash scripts/run_published_covariance.sh train-cohort --output results/published-covariance-v1-cuda-root --device cuda:0
# Supervisor checks the complete 27-state seal before any E prediction.
bash scripts/run_published_covariance.sh evaluate-cohort --output results/published-covariance-v1-cuda-root --device cuda:0
bash scripts/run_published_covariance.sh audit --output results/published-covariance-v1-cuda-root --device cuda:0
bash scripts/run_published_covariance.sh summarize --output results/published-covariance-v1-cuda-root --device cuda:0
.venv-published-covariance/bin/python plans/published-checkpoint-covariance/verify_real_report.py results/published-covariance-v1-cuda-root
```

All original [runbook](runbook.md) timeout, lock, immutable-artifact, pilot,
full-cohort seal and independent reporting requirements continue to apply. The
same grade-B checkpoint selection uncertainty and unknown historical project E
exposure apply, with the additional explicitly recorded partial first attempt.
