# Task-driven completion execution runbook

The implementation is ready for a **separately authorized CPU A01 seed-0 pilot**
under the frozen identity in [readiness.json](readiness.json). No new real
completion accuracy measurements exist. CUDA is unverified. This document does
not launch or authorize the pilot or cohort. Review the pilot before authorizing
the remaining tasks. No architecture, rank, optimization, masking or selection
settings may change after observing results.

Run commands from `/home/kalexu97/Projects/AGFL`. The declaration is
`configs/task-driven-completion.json`; it fixes nine participants, seeds 0/1/2,
five strategies, 27 tasks, 54 supervised completion fits and zero backbone fits.
All execution uses CPU, one thread, train/validation only. Historical inputs are
read-only. Full-input historical predictions must replay before each fresh task
can calibrate or fit any completer.

## Inspect and verify software

```bash
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --plan
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 04
```

Preflight is inventory-only; its `real_fitting_ready=false` is deliberate. It
does not replace original replay, independent software acceptance or pilot review.
The final worker and supervisor acceptance each passed 60 tests without skips.
The prior supervised-Tucker suite passed 25 tests. The raw completion-Transformer
suite passed 35 tests with one preexisting fixture error: its declaration-loading
test assumes the already completed historical output is empty. The unchanged test
passed in a fresh, byte-identical temporary config fixture. The raw failed log,
isolated pass and reproducing harness are retained under [validation](validation/).
No historical files or old tests were moved or edited to hide that failure.

## Synthetic smoke, completed-task resume and independent audit

Choose a fresh output path. The three commands below each start a new process;
the latter two operate on the completed smoke. Never reuse the path for changed
source, packages, configuration or synthetic input. The example path must be empty
or absent before the first command.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --smoke --output-dir /tmp/task-driven-completion-review-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --smoke --output-dir /tmp/task-driven-completion-review-smoke --resume-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion.audit --config configs/task-driven-completion.json --synthetic-smoke --output-dir /tmp/task-driven-completion-review-smoke
```

The audit has no fitting operation. It reconstructs the known smoke inputs and
source backbone, checks the stored numeric backbone against that source, loads
each saved selected completion state, forwards all 105 condition/repeat cells,
compares probabilities exactly, and independently recomputes balanced accuracy,
log loss and all five task contrasts. It does not accept the saved
`exact_selected_state_replay` flag as evidence of actual replay. Tests change
completion and backbone states, update their hashes and retain the flag: the
audit rejects both. Audit output is JSON on stdout and leaves task artifacts
unchanged. Completed-task resume verifies identity, structure, hashes and metrics;
use this additional audit for recomputation from inputs.

The final executed paths, exact commands, exit codes and frozen smoke identity
are in [execution.json](validation/execution.json). The smoke uses four synthetic
trials per partition and one supervised epoch, with all 30 calibration epochs.
Its reduced budget cannot establish real fitting duration or scientific benefit.
Synthetic evidence is excluded from real reports, which suppress cohort means
and intervals when real coverage is incomplete.

## Freeze identity before a separately authorized real pilot

Run the following comparison immediately before real execution. It checks all
hashed reused/new source, config bytes and resolved settings, installed packages,
Python version and historical manifests against the recorded real identity.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from inm.task_driven_completion.protocol import load_config, study_identity
cfg = load_config('configs/task-driven-completion.json', allow_existing_output=True)
expected = json.loads(Path('docs/task-driven-completion/validation/frozen-real-identity.json').read_text())
assert study_identity(cfg) == expected, 'Identity changed: stop and establish a fresh reviewed output identity'
print('Frozen real identity verified:', expected['study_id'])
PY
```

The frozen real study ID is
`1b2d20abcfd1bfa824b7dba5d77b38d5cfc5826ee820cd8943b1892aa0ec45ee`.
Its new output is `results/task-driven-completion-v1`. Source/config/package
changes require a new output identity and renewed checks. Do not alter existing
identity records, reuse incompatible artifacts or relax historical verification.
The supervisor verified all 27 historical identities, packages, splits and
checkpoint bundles; its original A01 full-input replay passed for both backbones.
Those separately attributed logs are preserved in validation. Every fresh real
task still performs its own verification and original full-input replay.

## Real phase 1: A01 seed-0 pilot, only after explicit authorization

Run task 0 once. Preserve stdout/stderr and all failed attempts. Do not retry a
failed or partial task or delete its output. Stop and review any failure.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --task-index 0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion.audit --config configs/task-driven-completion.json --task-index 0
```

The real audit reads original train/validation through the existing historical
readers into external temporary staging, verifies original full-input replay,
compares the saved backbone to its historical checkpoint and recomputes the
saved completion predictions. It performs no calibration or training and does not
write the completed study or historical inputs. This real audit path was not
executed during the coding session; only its synthetic path was exercised.

**Stop for pilot review.** Review task status and identity, original replay,
unchanged backbone/full-input predictions, paired masks and sample IDs, all 105
cells, finite solve/loss history, selected epochs including possible epoch 0,
independent audit, elapsed resources and negative as well as positive task
effects. A single pilot is not a cohort conclusion. Record the review before any
remaining task is launched. A change in settings is a new experiment, not a pilot
repair or a reason to reuse this output identity.

## Real phase 2: remaining cohort, only after pilot review and authorization

Recheck the frozen identity above. Then explicitly run remaining task indices
1–26. This command stops on the first failure and has no retries. It must not be
combined with the pilot command into an unattended launch.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
import subprocess, sys
for task_index in range(1, 27):
    subprocess.run([sys.executable, '-m', 'inm.task_driven_completion',
        '--config', 'configs/task-driven-completion.json', '--task-index', str(task_index)], check=True)
PY
```

If interrupted between tasks, an explicit repeated index may resume only a fully
completed, identity- and hash-verified task. Failed/partial tasks are never retried.
Keep failure evidence and resolve the execution decision separately.

## Real phase 3: audit every completed task and report

Only after the cohort has completed, independently audit every selected state,
then build the report. Each command stops on failure. Redirect audit stdout to
an external reviewed log directory if permanent real audit reports are needed;
do not put foreign files in the study output root.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
import subprocess, sys
for task_index in range(27):
    subprocess.run([sys.executable, '-m', 'inm.task_driven_completion.audit',
        '--config', 'configs/task-driven-completion.json', '--task-index', str(task_index)], check=True)
PY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --summarize-only
```

Reporting exits 2 for incomplete real coverage and withholds cohort estimates.
Review all five contrasts, equal-participant aggregation, negative effects and
variation. The primary exploratory screen requires at least 2 percentage points
and at least 6/9 positive participants; assess strong-control contrasts separately.
Intervals use 2,000 paired participant resamples and are unadjusted. Historical
and completion checkpoints both reused validation selection. These results cannot
provide independent confirmation, causal recovery or architecture superiority.
