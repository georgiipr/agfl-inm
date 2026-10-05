# Session 04 handoff: paired evaluation and reporting

Completed 2026-10-05. This session added the task runner, numeric artifacts,
resumable identity checks, public CLI and hierarchical validation reporting.
Changes remain within `inm/completion_transformer/`,
`tests/completion_transformer/` and this documentation directory. The frozen
configuration and all historical packages/results were left unchanged.

## APIs and behavior

- `study.initialize_output(cfg)` establishes or verifies the exact config,
  source, package and historical-manifest identity. It rejects output paths
  that overlap the repository, data or historical input roots, including
  resolved symlink paths, and refuses unowned files.
- `study.run_task(cfg, task_index)` prepares data through `replay.prepare_task`
  in a fresh task subdirectory, replays both historical models and verifies
  saved full-input predictions before fitting completion. It fits covariance
  and Tucker completion from training raw signals only, then evaluates the six
  backbone/strategy cells on paired deterministic masks. The original mask
  flags pass through every adapter. Full-input probabilities are gated against
  the original model and across all strategies before the task can complete.
- Each task stores probabilities, labels, ordered sample IDs, original Boolean
  masks, condition metadata, recomputable metrics, completion state/history,
  and SHA-256 hashes. Complete resume verifies every file and recomputes metrics
  from probabilities. Real resume and reporting also re-verify historical task,
  fit, checkpoint, prediction and history hashes. Failed or partial tasks are
  retained and never retried in place.
- `reporting.summarize(cfg)` checks all cell grids, deterministic mask semantics,
  pairing and metric recomputation. It reports per-repeat results and
  repeat-then-seed-then-participant aggregation, predeclared primary/secondary
  contrasts, the backbone difference-in-differences, participant bootstrap
  intervals, per-condition log loss, incomplete coverage and issues. Incomplete
  cohorts keep participant-level evidence but receive no cohort means or
  intervals. Synthetic tasks are always excluded from real evidence.
- `python -m inm.completion_transformer` provides lazy `--plan` and
  `--preflight`, explicit `--task-index`, `--smoke --output-dir`, resumable
  `--resume-smoke`, and `--summarize-only` modes. Planning imports only the
  standard-library protocol module. Unavailable preflight and incomplete real
  summaries return nonzero.

The fixed plan reports 27 tasks, six cells per task, no neural fits and 27
Tucker calibrations. Each cell contains 21 repeat/condition rows: one full-input
row and five repeats for each of four degraded conditions. The interaction is
reported as a difference-in-differences, without a synergy claim. The bootstrap
uses participants as the resampling unit (2000 resamples, seed 20261005).

## Verification executed

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/completion_transformer -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m compileall -q inm/completion_transformer tests/completion_transformer
git diff --check -- inm/completion_transformer tests/completion_transformer docs/completion-transformer
```

All 32 tests passed, including the 25 protocol/completion/adapter/replay tests
from sessions 01–03. The new tests cover numeric metric recomputation, mask
pairing and corruption, synthetic exclusion and partial reporting, dependency-
light CLI plan/preflight, a full 126-row synthetic smoke, resume verification,
changed identity, refusal to retry a failed task, and hand-calculated complete
cohort hierarchy/primary/secondary/interaction contrasts. A missing seed
fixture confirms that incomplete reports retain per-repeat rows while
suppressing cohort means and intervals; negative participant effects remain
visible. The only runtime notice
was PyTorch's existing `padding='same'` convolution warning for an even kernel.

An explicit CLI smoke completed at `/tmp/agfl-completion-smoke-04b`: it used
untrained synthetic backbones and one Tucker factor epoch, wrote 126 synthetic
evaluation rows, and produced a partial report with the synthetic task
excluded. This is a software check only; it is not a real study result.

## Limits and remaining work

No real completion factors were fitted and no real validation task was
evaluated. No historical input or result was modified. The real 27-task replay
still requires session 05 readiness review and a separately authorized run.
Smoke results cannot enter the real report. Validation checkpoints were
selected on the same participants, so any resulting intervals remain
exploratory rather than independent confirmation. Full-mask equivalence tests
are CPU checks; no CUDA execution was attempted.
