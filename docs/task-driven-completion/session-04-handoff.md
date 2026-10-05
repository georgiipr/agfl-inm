# Session 04 handoff: independent replay and execution readiness

Completed within the supervisor's 1,200-second session on 2026-10-05. Read root
`AGENTS.md`, project orientation, contract/session and all three accepted prior
handoffs/receipts. Changes stayed within the new package, its tests and its new
documentation. Prior handoffs, protocol/scientific settings, plans, runners,
configuration, historical code and results were preserved.

## Delivered files and behavior

- `inm/task_driven_completion/audit.py`: reusable `audit_task(cfg, task_index,
  synthetic_context=None)` API and explicit audit CLI. Validates current identity
  and complete artifacts, compares saved numeric backbone to its source, loads all
  selected completion states and independently recomputes 105 prediction cells,
  BA/log loss and five task contrasts. It does not call training/calibration or
  trust the saved replay assertion. Real mode uses strict historical readers with
  external temporary staging and original replay; only synthetic mode was run.
- `tests/task_driven_completion/test_integration.py`: seven substantive tests for
  separate-process fresh/resume/audit, changed-identity rejection, rehashed state
  and backbone tampering, audit without fitting/shared evaluation, unequal class
  and trial counts, negative effects and incomplete mean suppression.
- `docs/task-driven-completion/runbook.md`, `review.md`, `readiness.json`, this
  handoff and `validation/`: exact commands, final identities, logs, hashes,
  supervisor-attributed historical/regression evidence and a retained synthetic
  artifact archive. Runbook separates explicit pilot, review, cohort and reporting.

The audit intentionally supplements completed-task checksum/schema verification.
Tests demonstrate well-formed, rehashed altered states still pass the structural
verifier but fail actual predictions/source-backbone checks. It writes JSON to
stdout and leaves task artifacts unchanged. Its metrics use an independent
four-class recall mean and clipped categorical log loss, and task contrasts
average repeats then the four degraded conditions. It publishes no cohort means.

## Actual commands and outcomes

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/task_driven_completion -p test_integration.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 04
.venv/bin/python -m compileall -q inm/task_driven_completion tests/task_driven_completion
git diff --check
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --plan
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --smoke --output-dir /tmp/task-driven-completion-worker04-final-20261005T074155Z
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --smoke --output-dir /tmp/task-driven-completion-worker04-final-20261005T074155Z --resume-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion.audit --config configs/task-driven-completion.json --synthetic-smoke --output-dir /tmp/task-driven-completion-worker04-final-20261005T074155Z
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config /tmp/task-driven-completion-worker04-final-20261005T074155Z-changed-config.json --smoke --output-dir /tmp/task-driven-completion-worker04-final-20261005T074155Z --resume-smoke
```

Integration discovery passed **7 tests, no skips**, 16.124 seconds on its first
attempt. Mandatory worker acceptance passed **60 tests, no skips**, 23.793
seconds; supervisor separately passed **60**, 23.772 seconds. Compile and
whitespace checks passed. Plan declares 27 tasks, 54 supervised completion fits
and no backbone fits. Preflight exited 0, inventory complete, CPU one thread and
CUDA unverified; it correctly remains inventory-only with fitting readiness false.

After the final source change, fresh smoke, completed-task resume and independent
audit each exited 0 in a different process. Five strategies, 105 cells, two
synthetic supervised fits; all 105 saved selected-state probabilities replayed
exactly. Audit performed zero fits. Real reporting remained partial and excluded
the synthetic task. A separate changed config copy changed its name and retained
the output path; resume exited **2 as expected** with changed-identity rejection.
No real or original config was edited. Actual command arrays, codes and output
paths are preserved in `validation/execution.json`; expected failure stderr is
retained. The inherited even-kernel convolution warning remains informational.

The supervisor independently ran:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/supervised_tucker -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/completion_transformer -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python .session-runs/task-driven-completion/04/isolated_legacy_fixture.py
```

Supervised Tucker passed **25 tests**. The raw completion-Transformer suite had
**35 passes and one error**: a preexisting declaration test assumes the already
completed historical output is empty. The unchanged failing test passed alone
with a byte-identical declaration and expected paths relocated to a fresh temp
fixture; the loader and occupancy assertions stayed unchanged. Both raw failure
and isolated pass, plus its reproducible harness, are copied with `supervisor-`
prefixes. This is not an unqualified raw 36-test pass. The supervisor advised no
repeat was needed; the worker did not modify the older suite or historical output.

## Frozen identity and scientific limits

Source SHA256:
`fd2124d452a9da15eca6112981092cfb30298a5ce192b4fb2a714ad1b13a70e7`.
Real study ID:
`1b2d20abcfd1bfa824b7dba5d77b38d5cfc5826ee820cd8943b1892aa0ec45ee`.
Full source map, exact packages/Python/config and historical manifest hashes are
retained in `validation/frozen-real-identity.json`. Source was frozen before the
final smoke; documentation writes do not change this scientific identity.
Readiness JSON inventories evidence hashes and the separately frozen smoke
identity. The retained synthetic archive records its original absolute output
path and is archival evidence, not permission to resume under a relocated identity.

The supervisor had already verified all 27 historical identities, packages,
splits/checkpoints and original A01 full-input replay for both backbones. Copies
in validation retain explicit supervisor attribution. The worker did not repeat
real recording reads. No new real completion calibration, fitting, degraded
evaluation or accuracy measurements occurred. CUDA and real completion/audit
execution remain unverified. Readiness is for a **separately authorized A01
seed-0 pilot**, followed by independent audit and review before a cohort decision;
it establishes no empirical benefit. Preserve incomplete and negative results,
fixed settings, reused-validation limitations and all failed attempts.
