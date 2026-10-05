# Bounded CUDA execution repair 01

The v1 real pilot stopped before calibration/fitting on the full-input CPU/CUDA
gate. Diagnostic `prefit-diagnostic.json` isolates fused Transformer inference:
default maximum probability error 9.38e-5, versus 1.02e-6 without fastpath;
the latter passes the unchanged rtol1e-4/atol1e-6 gate. No completion outcomes
exist from the failed pilot. Original source is archived under
`.session-runs/task-driven-completion-cuda/repair-01/v1-source`; all failed
artifacts, configs, logs, and receipts are protected and must remain unchanged.

One fresh inherited-model worker, deadline 1200 seconds. Allowed modifications:
`inm/task_driven_completion_cuda/execution.py`, `protocol.py`,
`tests/task_driven_completion_cuda/test_gpu.py`, `test_protocol.py`.
Allowed new files: `configs/task-driven-completion-cuda-v2.json` and
`docs/task-driven-completion-cuda/repair-01-handoff.md` only.
Supervisor owns plans, acceptance, and session state. No real fitting.

Disable torch.backends.mha fastpath in scoped CUDA execution, restore caller
state on success/errors/nesting, assert and record the actual backend setting.
Declare this execution setting in FIXED and the new v2 config; keep all science,
cross-device tolerances and all original source untouched except the exact files
listed above. New config uses name/output task-driven-completion-cuda-v2.
Update synthetic tests to use v2 config and check actual fastpath disabling,
restoration and metadata enforcement. No other scientific or implementation edits.

Acceptance: three structural tests; GPU suite including deterministic repeated
fits and CLI smoke/resume/audit; supervisor separately checks original A01
checkpoint CPU/GPU prefit replay without fitting. Independent review must verify
all protection hashes and exact science equality excluding execution. Only after
passing, freeze v2 identity and run a new real pilot with new logs/output.
