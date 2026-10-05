# Bounded execution runbook

This runbook starts after session 05 software review. It separates metadata
readiness, the first real task, operational review, and any later cohort run.
The current session completed readiness checks only; it did not fit completion
factors or evaluate degraded inputs.

Run from the project root with the declared config:

```bash
.venv/bin/python -m inm.completion_transformer --config configs/completion-transformer.json --plan
.venv/bin/python -m inm.completion_transformer --config configs/completion-transformer.json --preflight
```

The plan must report 27 tasks, six backbone/strategy cells per task, zero
classifier fits, and 27 Tucker calibrations. Preflight is a byte-level input
inventory. It does not establish semantic compatibility; the task verifier
does that when each task starts.

After an operational review of software and available storage, the first real
operation is task index 0 (A01, seed 0):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.completion_transformer --config configs/completion-transformer.json --task-index 0
```

This task replays the original checkpoints, checks their saved full-input
validation probabilities, fits covariance and Tucker completion on its
training trials only, then evaluates the five paired availability conditions
on validation trials. It writes only under the declared new output identity
`results/completion-transformer-v1`. Inspect its `task.json`, completion state
and history, cell artifacts, and task checksums before considering more tasks.
Do not rerun a failed or partial task in place; preserve it and choose a new
study identity if a source or setting changes.

The pilot's operational review is a separate decision point. A cohort run
begins only after that review and explicit authorization. It consists of task
indices 1 through 26, one invocation per task, with no automatic retry. Preserve
all task outputs and run `--summarize-only` after execution or when checking
partial coverage. The report withholds cohort means and intervals unless all
27 real tasks verify. It averages mask repeats, then seeds within participant,
then participants equally. Its intervals are exploratory because the frozen
checkpoints were selected on these validation participants.

Never use test partitions, reselect checkpoints, tune completion settings,
change source after observing real outcomes, or infer a clean-input gain: full
input bypasses completion by design. Synthetic smoke artifacts are software
evidence and are excluded from real reports.
