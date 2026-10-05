# Session 03 handoff: historical checkpoint replay

Completed 2026-10-05. Replay verification is confined to
`inm/completion_transformer/`, synthetic checks to
`tests/completion_transformer/`, and this handoff to
`docs/completion-transformer/`.

## Implementation

Added `inm/completion_transformer/replay.py` and lazy replay exports from the
package initializer. `verify_historical_task(cfg, subject, seed)` treats
`report/evidence.json` as the candidate study manifest, verifies its complete
27-task inventory, the selected task and fit checksums, both result records,
history selection records, prediction array metadata, checkpoints and every
one of the 77 saved source checksums. It requires the explicitly reused data,
model, loader and training modules in the historical map and checks installed
package versions against the manifest. Paths are constrained to their declared
roots.

The verifier checks the original candidate preprocessing config by its exact
historical hash and matches its preprocessing settings to task provenance. It
uses only the baseline study's split and dataset metadata: their byte hashes,
split identity checksum, data fingerprint, ordered train/validation IDs,
channel order, and train-only normalization are checked against candidate task
metadata. It does not open baseline model files. Task, fit, checkpoint, split,
data and normalization identity mismatches fail with field-specific errors.

`prepare_task` validates historical lineage before calling the existing
read-only split/data helper. The helper may write its regenerated
`data_provenance.json` only to a fresh, empty task staging directory strictly
inside the new output root. Staging that overlaps historical inputs, sits
outside the new output, or already contains files is refused.

`restore_backbones(record)` rebuilds only `spatial_eegnet` and
`spatial_transformer` using the candidate restore factory and
`torch.load(..., weights_only=True)`. It verifies the full checkpoint metadata
against the fit record and returns frozen evaluation models. `replay_full_input`
compares both restored models' full-input validation probabilities, ordered IDs
and labels against the persisted predictions. It permits only CPU float32
rounding at `rtol=1e-5, atol=1e-7`.

## Verification

Executed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/completion_transformer -q
.venv/bin/python -m compileall -q inm/completion_transformer tests/completion_transformer
git diff --check -- inm/completion_transformer tests/completion_transformer docs/completion-transformer
```

All 25 tests passed, including the 17 session-01/02 checks. Eight replay tests
cover a historical-schema fixture, safe restore of both backbones, altered
checkpoint bytes, saved-source drift, regenerated manifest hashes around
split/data ID, normalization and sample-ID tampering, output-staging safety,
and matched/mismatched synthetic numerical replay. The only runtime notice was
PyTorch's existing `padding='same'` convolution warning.

Read-only readiness replay also succeeded for real A01/seed 0. The verified
prepared identity was `ad5503de7c09ad7979987e567f2a287b6a1b083a1c89af4a5a7e2dd6d6a24618`
with 163 training and 55 validation trials. Both saved backbones reproduced
all 55 validation predictions: maximum absolute probability error was
`1.49e-8` for spatial EEGNet and `2.33e-10` for spatial Transformer. The
temporary task provenance was written under a fresh temporary output root and
removed with that temporary directory.

No completion factors were fitted, no masks or degraded conditions were
evaluated, no checkpoint was reselected, and no historical input was written.
The remaining boundary is the planned next session's new-task execution; this
handoff establishes one real historical task's replay readiness, not full-cohort
readiness.
