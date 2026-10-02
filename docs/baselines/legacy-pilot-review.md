# Legacy A01/seed-0 pilot: ready for session 12

The legacy pilot in `results/inm-v2/artifacts/A01_seed_0` completed all 14
classifier fits. Its report has zero invalid records and zero recorded errors.
The study reports 14/378 fits because only the requested pilot was run; missing
cohort means are expected. Do not launch the remaining legacy matrix merely
to make that report complete.

## Diagnostic reader correction and verification

The session-09 reader expected `calibration_sha256` inside the calibration's
identity. Actual legacy calibration files store this digest as top-level
`cache_sha256`; result identities correctly use `calibration_sha256`. This
field-name mismatch falsely hid the valid shared-encoder score and all head
comparisons. The reader now maps the saved format to a common identity, while
still rejecting absent, conflicting, or mismatched calibration hashes.

The actual `calibration.pt` file hash matches `calibration.json` and all 14
result identities. The corrected diagnostic has no missing sources for the
pilot and identifies four matched baseline/core pairs (MHA/Performer ×
full/mixed). Selected validation scores agree with their history rows. All 64
original legacy files remain byte-for-byte unchanged. Thirteen diagnostics
tests pass, including real-format, missing-hash, conflict, and mismatch cases.
The previous reader and file checksums are archived under
`.session-runs/accuracy/legacy-diagnostic-review/`; its `stage-comparison.json`
records the detailed pilot diagnosis. No training was rerun.

## Validation evidence for planning

These are selected full-input validation balanced accuracies for one
participant/seed. Independently selected maxima are selection-biased diagnostics,
not independent evidence of a scientific effect. Test metrics were excluded
from the recommendations below.

| Stage / head | Full training | Mixed training |
|---|---:|---:|
| Shared encoder pretraining | 54.40% | Not applicable |
| MHA baseline | 53.16% | 52.75% |
| MHA tensor core | 54.53% | 60.03% |
| MHA tensor completion | 53.16% | 52.75% |
| MHA mean completion | 53.16% | 52.75% |
| Performer baseline | 51.24% | 49.31% |
| Performer tensor core | 51.24% | 49.31% |

The MHA full-input baseline is close to shared pretraining here, so this pilot
does not show a large loss specifically from retraining that head. The mixed
MHA core's higher validation score is a reason to retain it as a hypothesis in
a matched study, not to select a rank or claim improvement. Completion/mean
controls are needed to interpret missing-channel effects; full-input scores
alone cannot establish a robustness benefit. The new spatial EEGNet study has
a different feature source and must not serve as a matched tensor-effect control.

## Next command

The real baseline and legacy diagnostic prerequisites for optional session 12
are now present. Sessions 01–11 receipts still validate. Run:

```bash
cd ~/Projects/AGFL
bash scripts/run_accuracy_sessions.sh --from 12 --through 12
```

This coding session prepares and tests the smaller MHA-only tensor study in a
new output directory; it does not launch a real GPU experiment. It must reuse
the existing frozen feature source, solver, paired splits/masks, and selection
rules, with baseline/completion/core and explicit controls. Rank/ridge settings
remain predeclared; one participant's selected validation score is insufficient
to tune them. Use the completed session's `docs/baselines/tensor-followup.md`
for its exact pilot commands and validation-only decision rule.

The diagnostic source changed, so new experiments require fresh output identities.
Keep the completed legacy pilot and all baseline studies as original evidence.
