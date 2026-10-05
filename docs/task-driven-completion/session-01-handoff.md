# Session 01 handoff: fixed protocol and dependency-light provenance

Completed 2026-10-05 on `research/baseline-accuracy`, within the supervisor's
20-minute session window. Read root `AGENTS.md`, project orientation, the
task-driven-completion contract/session, and predecessor completion-transformer
session 01/05 handoffs and verified supervisor receipts. Existing source,
untracked work, historical artifacts, plans and orchestration were preserved.

## Files added

- `inm/task_driven_completion/__init__.py`: dependency-light public API exports.
- `inm/task_driven_completion/protocol.py`: fixed science, strict config loading,
  task planning, source/package/historical identity and read-only inventory.
- `inm/task_driven_completion/__main__.py`: plan/preflight only; no fitting mode.
- `configs/task-driven-completion.json`: nine participants, three seeds, five
  completion strategies, one frozen Transformer, CPU one thread, new output.
- `tests/task_driven_completion/test_protocol.py`: 14 synthetic/metadata tests.
- `docs/task-driven-completion/protocol.md`: scientific protocol, selection
  exception, identity and execution boundaries.
- `docs/task-driven-completion/session-01-handoff.md`: this handoff.

## Public APIs and downstream use

`protocol` exposes `FIXED`, `SCHEMA`, `ARM_IDS`, `HISTORICAL_ARM_IDS`,
`STRATEGY_IDS`, `LEARNED_STRATEGY_IDS`, `CONDITIONS`, `load_config`, `tasks`,
`plan`, `study_identity`, `inspect_inputs`, `digest`, `file_sha256` and
`validate_output_path`. Science lives in `cfg['protocol']`, with distinct
`tucker`, `covariance`, `training`, `training_masks`, `selection`, `execution`,
`historical`, `mask_policy`, `artifacts` and `evaluation` sections. Numerical
code should read these validated settings rather than introduce tunable options.

The normal plan is 27 tasks, **54 supervised completion fits**, **0 backbone
fits**, five strategies, and 21 evaluation rows per strategy/task. Learned
completion selection uses mean degraded validation BA over static/dynamic 16/6,
five repeats each, with lower degraded log loss then earlier epoch as ties;
epoch 0 is eligible and training runs at least 10 epochs before stopping.

`load_config` refuses scientific drift, incorrect scalar types (including bool
versus int), duplicate/unknown/missing JSON keys and occupied output by default.
All paths resolve relative to config location. `validate_output_path` checks
source/config/data/historical-root overlaps and existing output symlink
descendants; call it again immediately before future writes. An explicit
`allow_existing_output=True` permits read-only inspection only. Session 03
must implement actual identity/artifact verification before resume, and refuse
foreign, partial, failed and corrupted task output.

`study_identity` includes config bytes/resolved config, all root/agfl/inm Python
source (new package and reused dependencies), package/Python versions, original
candidate config and historical top-level/per-task/split metadata hashes.
The new package's later implementation changes will change this identity. New
source/config/packages requires a fresh output identity for real artifacts.

`inspect_inputs` reuses only the existing completion-transformer standard-library
file inventory. The existing `verify_historical_task` also loads prediction NPZ
arrays with NumPy; do not call it from dependency-light preflight. Later real
execution must call that strict historical verifier (both historical backbones)
and original full-input replay, then train completion only for Transformer.
The numerical replay adapter is `no_grad` and cannot train completers.

## Commands and actual results

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 01
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --plan
.venv/bin/python -m compileall -q inm/task_driven_completion tests/task_driven_completion
git diff --check
```

Final acceptance passed **14 tests, no skips** (0.579 s). Compilation and diff
whitespace checks passed. The public plan printed the counts above and
`real_fitting_ready=false`. Tests cover each fixed scientific leaf, cohort and
strict JSON types, config-relative paths, occupied/foreign output, overlaps in
both directions, protected repository and other historical results, symlink
parents/descendants, stable and changed config/source/package/history identity,
opaque-byte inventory and no readiness claim. Subprocess import guards prove
plan/preflight do not import Torch, NumPy, SciPy, scikit-learn or MNE. The CLI
rejects fitting flags, CUDA flags and incompatible modes.

The public real-path preflight was also invoked and its JSON reduced to a
console summary using this exact command:

```bash
.venv/bin/python - <<'PY'
import json
import subprocess
import sys
result = subprocess.run([sys.executable, '-m', 'inm.task_driven_completion', '--config', 'configs/task-driven-completion.json', '--preflight'], capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr or result.stdout)
inventory = json.loads(result.stdout)
print(json.dumps({key: inventory[key] for key in ('status', 'device', 'threads', 'cuda_verified', 'real_fitting_ready', 'historical_verification', 'original_full_input_replay', 'missing_packages', 'missing_recordings', 'missing_candidate_artifacts', 'missing_split_metadata', 'missing_historical_manifests')}, indent=2))
print('recordings:', len(inventory['recordings']), 'candidate tasks:', len(inventory['candidate_tasks']), 'split tasks:', len(inventory['split_metadata_tasks']))
PY
```

Preflight exited 0 with `inventory_complete`: 9 recordings, 27 candidate task
bundles (both historical backbones), 27 split metadata pairs, both historical
top-level manifests and all required packages present. All missing lists were
empty. It explicitly reported CPU one thread, CUDA unverified,
`real_fitting_ready=false`, `historical_verification=not_run` and
`original_full_input_replay=not_run`. Files were read as bytes for checksums;
no recording, checkpoint or prediction arrays were deserialized, and no study
output was created.

## Unresolved gates

Session 02 must implement the matched differentiable completers and frozen
backbone adapter and check numerical solves/gradients. Session 03 must implement
deterministic training, paired partition-disjoint masks, epoch-0 selection,
numeric artifact replay, identity-safe resume and hierarchical reporting.
Session 04 owns public numerical integration and regression checks. This
session's passing software checks establish neither model correctness nor
empirical benefit. No real calibration, fitting, degraded evaluation, original
prediction replay or test data access occurred. Real pilot and cohort remain
separate explicit operations after the required gates and pilot review.
