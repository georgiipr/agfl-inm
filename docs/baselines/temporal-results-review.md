# Completed temporal-head comparison

`results/baselines-temporal-v1` contains 108/108 valid fits. Rebuilding the
report returned success, with zero missing, invalid, failed, or excluded pairs.
All 108 checkpoint/history checksums and selected-history epochs were checked.

The predefined rule compares full-input validation balanced accuracy within
each training regime, averaging seeds within participant and then participants
equally. The completed cohort gives:

| Training regime | Flatten validation BA | Temporal validation BA | Temporal minus flatten | Participants with positive difference |
|---|---:|---:|---:|---:|
| Full | 60.86% | 57.08% | -3.77 percentage points | 3/9 |
| Mixed | 51.08% | 51.55% | +0.47 percentage points | 5/9 |

By that declared point-estimate rule, retain flatten for full training; temporal
has the nominal edge for mixed training. There is no tie requiring the log-loss
tie-breaker. The small mixed-regime advantage is not evidence of a statistically
established improvement. No confidence intervals or hypothesis tests were
computed, and no test score was used for this head choice. These outcomes do
not support replacing flatten with temporal for every regime.

The flatten control's full-input cohort scores reproduce the corrected baseline
cohort scores. The temporal model has fewer parameters and a different pooling
operation, so the experiment compares those complete heads; it cannot isolate
temporal convolution from the change in capacity or pooling.

## Next run for the original tensor research question

Update: the legacy pilot below is now complete. Read
[legacy-pilot-review.md](legacy-pilot-review.md) before proceeding to session 12.

At the temporal review, there were no legacy `results/inm-v2` outputs to diagnose. Before requesting
session 12's reduced tensor follow-up, produce one legacy A01/seed-0 pilot.
This fits the shared channel-local encoder and 14 classifier arms. It is a
diagnostic prerequisite, not an instruction to run all 378 classifier fits.

From a host terminal with GPU access:

```bash
cd ~/Projects/AGFL
.venv/bin/python run.py --config configs/study.json --task-index 0
```

The legacy entry point uses CUDA by default; it does not accept the baseline
launcher's `--device` flag. It writes into the separate `results/inm-v2` directory.
The configured `../ml` dataset path is resolved from the repository working
directory. The 27-task/378-fit legacy planner was checked before recommending
this pilot.

After the pilot finishes, rebuild its partial-cohort report:

```bash
.venv/bin/python run.py --config configs/study.json --summarize-only
```

Inspect the validation-stage evidence using
[legacy-diagnosis.md](legacy-diagnosis.md). Compare shared-encoder pretraining,
retrained baseline heads, and Tucker heads with matched calibration identities.
One participant can reveal operational issues but cannot establish an effect
or justify selecting ranks from favorable scores. Read that evidence before
running optional session 12, and keep the existing CNN and legacy feature-space
studies separate. No legacy training was launched during this review.

If the next research goal instead becomes cross-session benchmark accuracy,
the separate T/E study still requires official E-session labels; they are not
present under the dataset directory in this checkout. Do not infer those labels
from unknown GDF cue codes.
