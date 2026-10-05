# CUDA implementation handoff

Bounded implementation on 2026-10-05. Added files only under the authorized CUDA
package, CUDA tests, CUDA documentation and the new CUDA configuration. No real
fitting, real calibration or real degraded evaluation was launched by the worker.
The parent supervisor owns GPU execution, acceptance, preservation verification,
readiness evidence and subsequent pilot/cohort operations.

The old completion mathematics, CPU initialization/state readers and adapter are
imported read-only through small explicit re-export modules. CUDA protocol,
training, study, reporting and audit are explicit adapted copies; historical
runtime functions are not monkeypatched and CPU guards are not bypassed.
Scientific protocol equality is tested after removing the execution declaration.
The new schemas and default output are distinct from the CPU study.

`execution.scoped_cuda_rng` owns cuda:0 only, deterministic backend state and
cuBLAS initialization/environment, restores ambient state on errors, and fails
without CUDA. Study execution and audit transfer actual input/model/completion
tensors to CUDA. CPU calibration still runs all 30 original epochs and strict
original historical replay still precedes fitting. Float64 covariance parameters
and observed ridge solves are preserved. The new saved
`full-input-reference.npz` binds CPU/CUDA probabilities, labels and sample IDs.
Cross-device tolerance is fixed at rtol 1e-4 / atol 1e-6; within-device full-input
and selected-state replay remain exact.

Actual task hardware/runtime and source identity are recorded and checked on
resume, audit and reporting. Reporting excludes mismatched tasks with explicit
issues and withholds cohort estimates. The read-only audit loads selected states,
checks the numeric backbone against its immutable source, independently recreates
CPU full-input predictions, and reconstructs CUDA probabilities/metrics/contrasts.
It performs zero fits.

Worker commands already completed:

```bash
.venv/bin/python -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda.json --plan
.venv/bin/python -m unittest discover -s tests/task_driven_completion_cuda -p test_protocol.py -v
.venv/bin/python -m compileall -q inm/task_driven_completion_cuda tests/task_driven_completion_cuda
```

Plan: 27 tasks, 54 supervised completion fits, zero backbone fits. CPU structural
checks: **3 passed, no skips**; declaration equality, strict changed-setting
refusal, protected CPU output namespace, dependency-light imports, and source
coverage. Compilation passed. This is not CUDA readiness evidence.

The supervisor independently ran the required synthetic GPU checks outside the
sandbox on the NVIDIA GeForce RTX 3060 Laptop GPU. All **5 test groups passed,
no skips**, in **23.009 seconds**, on the first attempt. The suite deliberately
errors when no CUDA is present:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/task_driven_completion_cuda -p test_gpu.py -v
```

Five checks cover CPU/GPU outputs, input and completion-parameter gradients;
hidden NaN/Inf and observed-sample identity; frozen classification gradients and
full bypass; RNG/backend/thread/environment restoration on success/errors;
repeated one-epoch optimizer fits with changed arm order and ambient RNG; and
separate-process public smoke/resume/independent audit plus identity rejection.
Both gradient and output cross-device checks use rtol 1e-4 / atol 1e-6.

The unmodified supervisor log is retained at
`validation/supervisor-gpu-acceptance-attempt1.log`; its checksum and attribution
are in `validation/gpu-acceptance.json`. GPU output/gradient comparisons, exact
repeated fits and separate-process smoke/resume/audit all passed. The inherited
convolution warning and a Triton compilation macro redefinition warning did not
fail checks. No source/configuration was changed after that acceptance began.

Use the runbook for exact public commands and execution gates. The supervisor
owns preservation checks, final readiness and freezing the real identity before
its separately authorized pilot. The worker performed no real fitting.
