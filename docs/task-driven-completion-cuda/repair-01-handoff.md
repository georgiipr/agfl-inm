# CUDA fastpath repair 01 handoff

Completed bounded repair on 2026-10-05. The v1 pilot failed its prefit CPU/CUDA
probability comparison before any completion calibration or fitting. The preserved
supervisor diagnostic isolates fused Transformer inference: maximum error
9.381771087646484e-05 with the default path and 1.0132789611816406e-06 with
fastpath disabled. The latter passed the unchanged rtol=1e-4, atol=1e-6 gate.
These numbers come from `.session-runs/task-driven-completion-cuda/prefit-diagnostic.json`,
not new worker measurements or completion outcomes.

`scoped_cuda_rng` now saves the caller's MHA fastpath flag, disables it throughout
the scoped execution (including enclosed CPU replay), and restores it in `finally`.
Nested scopes preserve the enclosing state. `execution_metadata` reads the actual
backend flag, rejects enabled fastpath, and records `mha_fastpath_enabled: false`.
Existing task resume, audit and report comparisons therefore enforce this field.
The fixed protocol and new configuration declare the same setting.

The new config uses name `task-driven-completion-cuda-v2` and output
`results/task-driven-completion-cuda-v2`. Artifact schemas remain unchanged;
configuration and source hashes establish the new study identity. The v1 config
is untouched and is rejected by the updated strict declaration because it lacks
the required execution field. All scientific settings and cross-device tolerances
are unchanged, including CPU calibration, masks, factor mathematics, training,
selection and cohort sizes.

Exactly four existing files changed:

- `inm/task_driven_completion_cuda/execution.py`
- `inm/task_driven_completion_cuda/protocol.py`
- `tests/task_driven_completion_cuda/test_gpu.py`
- `tests/task_driven_completion_cuda/test_protocol.py`

Exactly two files added:

- `configs/task-driven-completion-cuda-v2.json`
- `docs/task-driven-completion-cuda/repair-01-handoff.md`

Worker validation completed:

```bash
.venv/bin/python -B -m unittest discover -s tests/task_driven_completion_cuda -p test_protocol.py -v
.venv/bin/python -B -m inm.task_driven_completion_cuda --config configs/task-driven-completion-cuda-v2.json --plan
```

All three structural tests passed, no skips, in 0.106 seconds. The plan remains
27 tasks, 54 supervised completion fits and zero backbone fits. Checks establish
exact science equality with the CPU declaration and exact v1/v2 configuration
equality after only name/output/fastpath changes; strict rejection of altered
science, enabled or non-Boolean fastpath, and the old execution declaration;
dependency-light planning; and reused/new execution source identity coverage.
AST parsing and trailing-whitespace checks passed for the four changed Python
files. A byte comparison against the archived v1 source confirmed exactly the
four allowed existing-file changes and preserved the other archived files,
including the original config.

The updated GPU suite uses v2 and adds both ambient fastpath states, successful
and exceptional nested scopes, actual-backend metadata rejection, and rejection
of enabled/missing fastpath metadata on completed synthetic task resume and
independent audit. Existing output/gradient comparisons, hidden-value invariance,
frozen backbone checks, repeated short fits with changed ambient RNG/arm order,
and fresh CLI smoke/resume/audit remain included. It now has six test methods.

GPU acceptance remains for the supervisor outside the sandbox:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests/task_driven_completion_cuda -p test_gpu.py -v
```

The worker ran no GPU tests, real calibration, fitting, degraded evaluation or
historical checkpoint replay. Supervisor acceptance, broader protection hashes
and real checkpoint prefit replay remain required before freezing v2 or starting
a new pilot. No plans, acceptance scripts, session state, prior logs, receipts or
results were edited. Source/config changes are finished at this handoff.
