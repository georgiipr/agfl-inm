# Session 05-R1 handoff: execution admission repair

**Status: bounded execution repairs implemented; supervisor source review is the
remaining gate.** This session ran synthetic checks only. It performed no real
covariance fitting and scored no official E trials. Native-checkpoint CUDA
fit/save/replay remains session 06 work.

The public initializer now lazily preserves its original data API, so the plan
command remains dependency-light. Study identity binds every Python module,
launcher, declared plans and audit recipe, configuration, the exact required
official source/checkpoint hashes, all 18 MAT byte hashes, complete installed
package versions, and JSON-canonical runtime/device flags. Frozen optimizer,
split, normalization, covariance and exact mask-key settings are declared in
the configuration.

Evaluation runs as 27 independently bounded subject/seed tasks. Each task has
33 expected cells, a task receipt and an independent replay audit receipt.
Incomplete tasks fail closed on resume; complete tasks replay before reuse.
The final audit checks exact task and cell coverage, task-row equality, source
and artifact hashes, native full-input references, and full-input equality
across arms and covariance seeds. It hashes the complete immutable output tree
so report generation can verify probability bytes without rebuilding all 891
cells. Writes reject symlinked roots and nested paths. Fresh real output names
use direct `results/published-covariance-*` directories after any identity
change.

The launcher preserves start/end times, command, stdout, stderr and exit code
for each bounded child. It audits each fit before the next fit, gates the 24
remaining fits on the reviewed pilot, and separately bounds T references and E
evaluation/audit tasks. Child failures propagate to the launcher. Reports
include cohort condition metrics and degradation, participant effects,
selected epochs and validation scores, epoch-zero and budget counts, native
trial identity hashes/counts, frozen classifier state hashes, parameter
counts, and separately measured completion and classifier probe latency.
Synthetic T and E fixtures have distinct trial identities and remain labeled
as generated data.

Checks completed against the final source include:

- `acceptance_execution.py`: complete synthetic lifecycle, immutable reuse,
  nested symlink rejection, incomplete/duplicate/foreign cells, probability
  byte tampering, report tampering, partial task resume and pilot marker reuse
  — passed (79.7 s).
- `acceptance_execution_faults.py` — 3 passed; `acceptance_identity.py` — 2
  passed; `acceptance_launcher.py` — 2 passed;
  `acceptance_launcher_sequence.py` — passed; `acceptance_reporting.py` — 2
  passed; `acceptance_native_reference.py` — generated CPU input parity passed
  with maximum probability error 0.0.
- Package unit suite — 19 passed; dependency-light plan — 27 covariance fits,
  0 backbone fits, 81 arm evaluations; Python compilation and Bash syntax —
  passed.

No official cohort result or CUDA readiness claim follows from these checks.
The worker environment exposed no CUDA device during CPU acceptance. Preserve
the real preflight, GPU allocation check, and native-checkpoint CUDA
fit/save/replay as separate execution gates.
