# Session 05 handoff: integration and readiness

Completed 2026-10-05 within `inm/completion_transformer/`,
`tests/completion_transformer/`, and `docs/completion-transformer/`. The frozen
config and historical experiment packages/results were preserved. Final
software and readiness identity is recorded in [readiness.json](readiness.json).

## Integration changes

Task output now stores complete primitive covariance and Tucker state plus the
state settings. `completion_state_arrays` serializes fitted flags, sample and
seen counts, covariance second moment, Tucker factors and fit steps. The public
`load_completion_state(path, history, expected_epochs, expected_settings)`
loads with pickle disabled and validates schema, settings, scalar types, exact
array dtypes and dimensions, finite values, fitted flags, Tucker epoch/history
agreement, factor norms, and covariance/count validity. Completed-task resume
and report validation perform these numeric checks after byte checksums.
Recomputing checksums cannot hide a NaN factor or incorrect Tucker rank.

The reporting fixture now uses meaningful synthetic state. Tests exercise
completion equality after NPZ save/load, incorrect dimensions, non-finite
state, fitted flags, history epoch counts and numeric diagnostics, and
wrong-rank state after recomputing the artifact checksum. CLI preflight tests
use intentionally missing input paths, expect a nonzero exit, and verify the
planning and preflight paths leave Torch, NumPy and SciPy unloaded.

## Verification executed

The complete package suite passed 36 tests on CPU. A public CLI synthetic
smoke wrote 126 evaluation rows and returned a partial report with its
synthetic task excluded. Compilation and `git diff --check` passed. The public
plan reported 27 tasks, zero classifier fits and 27 Tucker calibrations; real
preflight reported all recordings, candidate artifacts, split metadata and
required packages present. The exact outputs and logs are in `validation/`.

Read-only historical checks verified all 27 task metadata records and matched
the saved A01T.gdf through A09T.gdf hashes against current recording bytes. A01
seed 0 was prepared using only its original training and validation split.
Both historical frozen models reproduced all 55 saved validation
probabilities; maximum absolute errors were `1.49e-8` for spatial EEGNet and
`2.33e-10` for spatial Transformer. The replay used data identity
`ad5503de7c09ad7979987e567f2a287b6a1b083a1c89af4a5a7e2dd6d6a24618` and split
identity `45b40a60ab33a5410f166db56e491a2e7e2a08c7e365a50dbfe3acac280daab1`.
Temporary replay staging was removed after comparison.

## Readiness boundary

This is software and pilot readiness for operational review, not a real
completion result. Synthetic completion factors were fit only for the CLI
smoke. No completion factors were fitted on real EEG, no degraded real
condition was evaluated, and no new accuracy measurement exists. No CUDA
execution was attempted. The pilot and cohort steps remain separate explicit
operations described in [runbook.md](runbook.md); a cohort run follows only
after pilot operational review and explicit authorization. Preserve these
readiness artifacts and use a new output identity if code, packages, or config
change.
