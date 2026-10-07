# Bounded operational runtime-failure diagnosis

Read CONTRACT.md, ACCEPTANCE.md, the accepted 06 handoff/runbook, and immutable
failure `.session-runs/published-checkpoint-covariance/launcher/20261007T010348Z-215256-A05-s1-E-evaluate.stderr`.

Exactly one fresh gpt-6-luna high worker, 20-minute deadline. Supervisor owns
plans, acceptance, receipts, runtime changes and all real operations.

Observed failure: evaluation exited 134 while opening libdevice.10.bc through
`tensorflow/python/platform/../../libtensorflow_framework.so.2/../../nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc` (Not a directory).
All 27 fits and 13 E task audits completed; A05-s1 has zero task cells and no task
receipt. No cohort report exists. Preserve all artifacts and do not inspect
accuracy outcomes to guide any repair.

Worker write scope ONLY `docs/published-covariance/runtime-failure-handoff.md`.
Read-only inspection of local runtime/source/logs and contract is allowed. No
source/runtime/config/launcher edits, installs, GPU calls, fitting, E predictions,
worker spawning, artifact writes, or retries. Never mutate outputs.

Deliver: concrete diagnosis from local package files/logs; exact minimal proposed
runtime repair and generated-input verification commands; effect on identity;
whether unchanged-identity resumption is scientifically/technically legitimate;
if fresh identity required, distinguish model/optimizer changes from compiler
asset discovery and state precisely which preserved artifacts can be used as
read-only evidence. Do not invent compatibility or bypass identity checks.
Assess a deterministic generated-input XLA math probe and native all-nine
forward/backward smoke that can reliably establish library/ptxas discovery.
Prefer no package download: cuda_nvcc/bin/ptxas and nvvm/libdevice/libdevice.10.bc
are already installed. Explain uncertainty if crash is nondeterministic.

Supervisor acceptance: handoff evidence, protected hashes, no writes except
handoff; independently execute any accepted synthetic runtime check in fresh
paths and preserve exits. Scientific settings stay frozen. Prior partial E
exposure must remain disclosed in any successor output.
