# Session 03 handoff: deterministic execution and verified artifacts

Completed 2026-10-05 within the bounded session. Read the project orientation,
contract, session specification, predecessor handoffs and accepted receipts.
Only the seven session-authorized files were changed:

- `inm/task_driven_completion/training.py`
- `inm/task_driven_completion/study.py`
- `inm/task_driven_completion/reporting.py`
- `inm/task_driven_completion/__main__.py`
- `tests/task_driven_completion/test_training.py`
- `tests/task_driven_completion/test_study.py`
- `docs/task-driven-completion/session-03-handoff.md`

No real recording or historical checkpoint arrays were read by this worker.
There was no real calibration, completion fitting, degraded evaluation or test
access. Historical metadata bytes are included by the existing identity code.
No prior scientific code, config, plan, receipt or historical artifact was changed.

## Implemented behavior and APIs

`training.fit_completion(backbone, strategy, completer, train, validation,
subject=..., seed=..., synthetic=False, budget=None)` runs supervised completion
only. It returns initial/selected numeric states, history, selected epoch,
settings, training masks and training orders. `train` and `validation` expose
`raw`, `labels`, and unique, disjoint `sample_ids`. The fixed real budget is Adam
.001, batch32, max100/min10/patience20, clip1, CE + .1 hidden reconstruction +
.0001 initial-parameter anchor. Epoch zero is eligible; selection averages all
20 degraded validation evaluations and uses log loss then earlier epoch to break
ties. Full batches skip optimizer updates. Every completed epoch checks exact
full-input invariance; training checks all frozen backbone tensors and gradients.

`training_schedule`, `validation_bank`, `selection_key`, `evaluate`, and
`training_settings` are independently usable. Per-example mask choices and batch
permutations derive from stable task/epoch/sample keys, never arm or ambient RNG.
Training uses `make_mask_bank(partition='train')`; validation uses its disjoint
partition. Both learned arms share schedule prefixes even if stopping differs.
Model/training scopes restore Python, NumPy, Torch RNG and CPU thread counts on
success and failure. Only synthetic calls may inject a three-field positive
`maximum_epochs`/`minimum_epochs`/`patience` budget, recorded with the task and
included in its synthetic-context digest. Calibration still uses all 30 epochs.

`study.synthetic_context()` creates four synthetic train and four validation
trials, an untrained historical Transformer, and an explicit one-epoch budget.
`study.run_task(cfg, task_index, synthetic_context=None, strategy_order=None)`
executes all five strategies; arm-order overrides are synthetic-only. Real mode
calls the read-only historical preparation/verifier, **original full-input replay
of both backbones before any calibration**, and original backbone restoration.
Only the Transformer participates in the new completion experiment. This real
path is implemented but was not executed by this worker.

Task artifacts include numeric backbone snapshot, common Tucker/covariance
initialization, all 30 factor-history rows, initial and selected completion states
for every arm, learned histories and exact training masks/orders/IDs/labels, and
105 numeric prediction cells with masks/IDs/labels. Saved selected states are
loaded from NPZ with `allow_pickle=False`; all 21 predictions per strategy must
reproduce exactly. Full predictions match the unchanged backbone across arms.
Tasks persist source/config/package/history identity and every artifact checksum;
real resume additionally reruns strict historical verification and hashes original
fit/task/checkpoint inputs and the original T recording. Failed/partial/foreign
outputs, changed identities or changed synthetic data/backbone/budgets are refused.
No partial retry or optimizer resumption is implemented or authorized.

`reporting.verify_task_artifacts` checks required artifact names, primitive state
schemas, common initial states/anchors, factor-history numeric fields and epoch
sequence, backbone snapshot shapes/dtypes, exact 105-cell coverage, labels and IDs,
probability validity, recomputed BA/log loss, deterministic validation masks,
full-input invariance, selected history metrics, early stopping and update counts,
and regenerated paired training schedules. `summarize(cfg)` checks current identity
and historical hashes before including real evidence. `aggregate` averages repeats,
then seeds within participant, then participants equally. It preserves per-condition
BA/log loss, repeat/seed/participant variation, negative participant effects and
partial rows. Even an explicit `complete=True` cannot publish cohort means for an
incomplete declared task grid. Public real reports require all 27 tasks, exclude
synthetic evidence, and provide the fixed five contrasts and paired participant
bootstrap (2000 draws, seed20261005). The primary screen is separate from comparisons
to the stronger controls. Summary JSON, CSV tables and Markdown state the validation
reuse and lack of independent confirmation.

CLI modes now include dependency-light `--plan` / `--preflight`, explicit
`--smoke --output-dir PATH [--resume-smoke]`, real `--task-index INDEX`, and
`--summarize-only`. The latter returns exit 2 for incomplete real evidence.
Smoke always excludes its evidence from the real report. CUDA is not offered.
The public preflight remains inventory-only and never claims fitting readiness.

## Commands and actual outcomes

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 03 > /tmp/task-driven-completion-worker03-acceptance.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --smoke --output-dir /tmp/task-driven-completion-worker03-final-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --smoke --output-dir /tmp/task-driven-completion-worker03-final-smoke --resume-smoke
.venv/bin/python -m compileall -q inm/task_driven_completion tests/task_driven_completion
git diff --check
```

Mandatory acceptance passed **53 tests, no skips**, in 6.988 seconds. The 17 new
checks cover actual repeated short optimizer fits with reversed arm order and
changed ambient RNG, partition-disjoint per-ID masks, full-batch skipping and
epoch-zero selection, exception-state restoration, exact artifact resume,
changed source/budget rejection, checksums and recomputed metrics, rehashed mask
and calibration-history corruption, incomplete cells/selected epochs, failed and
partial tasks, original replay failure before calibration, hierarchical negative
effects, bootstrap repeatability and incomplete/synthetic suppression.

Final fresh smoke and resumed smoke both exited 0 with **105 cells, five strategies,
two supervised completion fits**, `status=complete`, and one synthetic task excluded
from the deliberately partial real report. The earlier development smoke under
`/tmp/task-driven-completion-worker03-smoke` predates final source changes and must
not be resumed under the final identity. Compile and whitespace checks passed.
The inherited even-kernel convolution emits its existing PyTorch warning.

## Remaining gates

Supervisor independent acceptance and session 04 public integration/regression
checks remain. Real execution is a separate explicit operation; no software smoke
establishes empirical benefit or authorizes real fitting. Original replay remains
mandatory on every fresh real task, and real pilot review precedes cohort execution.
CPU one thread is the implemented path; CUDA remains unverified. No completed real
study evidence exists from this implementation session.
