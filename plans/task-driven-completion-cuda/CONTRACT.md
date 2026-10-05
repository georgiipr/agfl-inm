# CUDA execution follow-up

User explicitly authorized CUDA experiments on 2026-10-05. This supersedes the
CPU-only execution restriction for this NEW CUDA successor, not prior artifacts.
Preserve all previous source, configs, plans, receipts and results unchanged.
A separate package inm/task_driven_completion_cuda/, tests/task_driven_completion_cuda/,
configs/task-driven-completion-cuda.json and docs/task-driven-completion-cuda/ owns
this execution adaptation. Supervisor owns this plan directory and session state.

Read plans/task-driven-completion/CONTRACT.md and prior handoffs. Same five arms,
27 tasks, 54 fits, all architectures/ranks/initialization/training/masks/selection/
metrics/budgets remain fixed. Calibration stays CPU one thread to preserve exact
paired initialization. Completion optimization, validation selection and final
predictions use cuda:0. Strict CPU historical checkpoint replay still precedes
all calibration; verify CUDA full-input probabilities against CPU with fixed
rtol1e-4/atol1e-6 before fitting. Within-device full bypass and selected-state
replay remain bitwise exact. CPU/GPU cross-device equality is not assumed.
New output results/task-driven-completion-cuda-v1, distinct schema/name/execution
metadata. No reuse of CPU smoke/scientific outputs under CUDA identity. Adding
new Python source changes the broad source digest; preserve old source snapshots
and receipts as historical evidence, never rewrite them to appear current.

Deterministic CUDA execution: CUBLAS_WORKSPACE_CONFIG=:4096:8 set before CUDA
initialization; deterministic algorithms enabled with errors, cuDNN benchmark
false/deterministic true, TF32 disabled; no AMP. Scope/restore Python/NumPy/CPU
and selected GPU RNG, backend flags, thread count and environment on exit/errors.
Do not seed unrelated GPUs. Record actual device, GPU name/capability, CUDA/cuDNN/
Torch/packages/source/config identity. CUDA missing is a hard error; no CPU fallback.
Keep covariance free parameters and observed ridge solves float64; do not cast
whole adapters to float32. Original source and model modules imported read-only.
Favor a small explicit new execution implementation; no monkeypatching historical
runtime functions to evade CPU protocol guards. Shared pure helpers may be reused.
Source digest must cover every reused/new execution source.

One bounded implementation worker, 1200 seconds. Allowed writes only the NEW
package/tests/config/docs above. Synthetic GPU checks outside sandbox require
escalation; workers can implement CPU structural checks then supervisor executes
GPU acceptance. Meaningful numerical/gradient GPU checks vs CPU with declared
tolerances, repeated short fits with changed ambient RNG and arm order, frozen
backbone/hiddenNaN/paired masks, fresh CLI smoke/resume and GPU audit required.
Do not run real fitting inside the coding session. No scientific tuning.

After independent acceptance and preservation checks, supervisor separately runs
real A01 seed0 pilot on CUDA with process timeout, logs and no automatic retries.
Audit selected states and inspect operational correctness, finite histories,
paired masks/IDs, original and CUDA full-input replay, exact GPU checkpoint reload,
and resource use. Pilot review is operational, never an accuracy-based gate or
settings tuning. User authorization extends to the real experiment sequence:
a passing recorded pilot review permits remaining tasks one at a time, each
with timeout, independent audit and receipt. Stop on errors/timeout; preserve all
attempts. New fitting authorization is already supplied; do not ask again.
Reporting must preserve negative/incomplete results and validation-only limits.
