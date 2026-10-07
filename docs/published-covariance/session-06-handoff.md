# Session 06 handoff: integration and readiness

**Status:** native synthetic CUDA integration and the independent common
acceptance runner passed. No real covariance fit or official E prediction was
run.

The public `smoke --native --device cuda:0 --output PATH` now writes generated
smoke evidence into an exclusive directory outside `results/`. It runs the
actual pinned native classifier for A01–A09, checks finite nonzero input
gradients and frozen state, and performs one generated T-only A01 covariance
training epoch through `train_covariance(..., synthetic=True)`. The smoke saves
the selected 253-value covariance state, reloads it, then compares exact native
GPU probabilities. It also saves the generated values, boolean channel mask,
initial/selected covariance states, gradients and actual training history. Output
paths reject symlink components and an existing output directory.

The independent supervisor run is recorded in
`.session-runs/published-checkpoint-covariance/06/native-cuda.json` (exit 0,
25.316 seconds). It invoked the public Bash launcher and
`plans/published-checkpoint-covariance/acceptance_cuda.py` on the NVIDIA GeForce
RTX 3060 Laptop GPU. The checker confirmed native forward/backward for all nine
subjects, GPU placement, finite nonzero covariance and input gradients, unchanged
frozen model and BatchNorm bytes, a real covariance update with parameter delta
norm 0.015874, and exact saved-state probability replay plus its independent
NumPy completion solve. Epoch zero won validation selection; this is allowed,
and the training history still records the nonzero update. The checker reports
zero real fits and zero real E predictions.

The human report now includes the covariance's 253 free parameters alongside the
115,172 frozen backbone parameters. `dependencies-lock.txt` is the exact
`pip freeze --all` from `.venv-published-covariance` (Python 3.11.15,
TensorFlow 2.15.1); an independent comparison against the installed freeze
matched. The updated [runbook](runbook.md) gives the dependency-light plan,
preflight, native synthetic smoke, native T references before the pilot, pilot
review, remaining fit/seal, E task evaluation/audit, final report and independent
numeric verification commands. It also gives the complete synthetic lifecycle,
output identity rules, process timeouts and current resource estimates.

Checks completed by this worker:

- `-m unittest discover -s tests/published_covariance -v` — 19 passed.
- `python3 -I -S plans/published-checkpoint-covariance/check_plan.py` — 27
  covariance fits, 0 backbone fits, 81 arm evaluations.
- `bash -n scripts/run_published_covariance.sh` and Python compilation for the
  CLI and new smoke helper — passed.
- Runtime package lock comparison — exact match.
- Protected inventory comparison — all 13,636 files from the accepted 05-R1
  snapshot match, except the two explicitly supervisor-owned acceptance updates
  (`acceptance.py` and `acceptance_execution.py`). No other protected file
  changed.
- Supervisor `plans/published-checkpoint-covariance/acceptance.py` — exit 0 in
  178.307 seconds, recorded in
  `.session-runs/published-checkpoint-covariance/06/common-acceptance.json`.
  This includes the independent data, classifier, covariance, training,
  reporting, launcher-fault and synthetic lifecycle checks.

Do not interpret the readiness checks as an accuracy result. Real pilot runtime
and peak memory still need to be measured before the remaining cohort fits, and
the supervisor must record its proceed/stop decision before starting any real
operation. Preserve the exploratory grade-B checkpoint lineage and unknown
prior E-outcome exposure in all reports.
