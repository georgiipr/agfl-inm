# Authorized real execution — 2026-10-05

The user explicitly requested running the experiments with the
sequential-session-watchdog skill after software implementation and review.
This authorizes the real pilot and, after a successful operational review,
the remaining predeclared cohort. No new neural training or setting search.

Run task 0 (A01/seed0), wait for completion, verify its artifacts and independently
recompute metrics, then record an operational proceed/stop review. Subsequently
run indices 1–26 one at a time, requiring the preceding review before each launch.
Finally summarize and independently recalculate the paired cohort contrasts.
Negative pilot scores are not an operational failure or a reason to tune settings.

CPU execution uses OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Each task has a
1200-second GNU timeout with a 15-second kill grace period. Execution and checking
share the workspace lock. No automatic retry after failure or timeout; preserve
partial outputs. The active supervisor checks progress at intervals of at most
60 seconds and does not leave an unattended run behind.

Before and after each task, verify the frozen readiness identity and hashes of
1,411 protected source and historical artifact files. Verify all 126 numeric
condition/repeat/backbone/strategy records per task, saved completion state,
original-checkpoint provenance, paired masks and full-input equality. Recompute
balanced accuracy and log loss directly from saved probabilities and labels.
The supervisor examines the resulting per-task summaries before advancing.

Logs and review receipts are under
`.session-runs/completion-transformer/experiments/run-20261005/`; scientific output
is `results/completion-transformer-v1/`. The existing source, fixed configuration,
30-epoch Tucker calibration and original selected checkpoints remain unchanged.
