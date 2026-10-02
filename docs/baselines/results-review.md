# First complete baseline study: results and next steps

The corrected replication is now complete. See
[the corrected-cohort review](corrected-results-review.md) for current results
and the next optional coding session. The findings below describe the original
study and the repairs that motivated its replication.

Reviewed `results/baselines-v1` on 2026-10-01. All 27 participant/seed tasks
and 108 fits are present. The repaired report accepts every record: no missing,
invalid, failed, or excluded neural pairs. The protocol is participant-specific,
within T session, with three seeds and nine participants. These scores do not
establish a cross-session benchmark or SOTA result.

## Measured outcomes

Balanced accuracy, in percent. Seeds are averaged within each participant, then
participants receive equal weight. Validation values are measured at selected
checkpoints and therefore contain selection optimism. Test values below are
descriptive evaluation results, not configuration-selection evidence.

| Arm | Full validation | Full test | Mean degraded test |
|---|---:|---:|---:|
| EEGNet reference / full training | 58.01 | 48.22 | 33.95 |
| Mask-conditioned EEGNet / full training | 55.59 | 48.94 | 32.91 |
| Mask-conditioned EEGNet / mixed training | 48.81 | 42.00 | 34.71 |
| Covariance classifier / full training | 56.12 | 56.35 | Not evaluated |

The degraded column is an additional descriptive average giving each of the 12
loss conditions equal weight, after the report's repeat/seed/participant
averages. It is not a predeclared epoch-selection score or a significance test.
Classical coverage is full-input only, so it cannot be ranked for robustness.

For the reference, clean training balanced accuracy at the selected checkpoint
averages 84.13%, versus 58.01% on validation. The corresponding numbers are
86.83%/55.59% for the full-trained masked model and 74.02%/48.81% for mixed
training. These gaps warrant investigating generalization and the training
protocol before adding a larger head. They do not identify a single cause.

Using only the five allowed validation banks (full, random-static/dynamic-random
at 16 and 6 channels), the existing checkpoints average 39.61%, 37.43%, and
40.00% for the three neural arms respectively. These checkpoints were selected
by full-input validation; this is not an experiment with robust selection.
Mixed training's small descriptive advantage on that allowed validation
aggregate accompanies a substantial full-input validation cost. Neither test
scores nor 11-channel/spatial validation conditions should drive the next model
selection decision. No participant-level uncertainty intervals were computed.

## Two defects found during cohort review

1. **Reporting identity mismatch.** The reporter compared prepared-data hashes
   across seeds. Those hashes include the seed, split and train-fitted
   normalization, so rejecting their differences incorrectly excluded 72 fits.
   The corrected reporter verifies each hash against `dataset.json`, then
   compares the underlying loaded-dataset fingerprint across seeds. All nine
   participants have one consistent underlying identity and three distinct
   prepared identities. No result JSON or metric was rewritten.
2. **Initialization was not controlled by the task seed.** `fit_classifier`
   seeded training after `EEGNetClassifier` had already initialized its weights.
   Initial weights could depend on process RNG state or preceding arms. Model
   construction is now seeded separately before every neural arm. A repeated
   synthetic fit with the same task seed but different ambient RNG states
   failed before the fix and reproduces checkpoint tensors exactly after it.

The first study's saved models and measurements remain usable as records of
those executions. Training them again from the declared seed alone was not
reliable. This does not demonstrate that the initialization bug caused the
low accuracy; the corrected replication must measure its effect.

All 54 baseline tests now pass, including data preparation/report integration
across seeds, rejection of real dataset changes, metadata-tampering checks,
and training reproducibility. Checksums for all 379 original experiment files
outside `report/` remain unchanged. Checkpoint/history/classical-state hashes
and selected epochs were checked for all 108 fits. Original reports and changed
source files are archived in `.session-runs/accuracy/reporting-seed-fix/`;
the archived Python files match the original result source hashes.

## Recommended next experiment

Replicate the unchanged baseline design with corrected initialization before
sessions 11–12. `configs/baselines-reproducible.json` differs from the original
JSON only in its study name and output directory. It retains all subjects,
seeds, arms, preprocessing, training settings, and selection policy. Source and
config identity changes require separate outputs; never overwrite or mix the
two studies.

From the repository root in a terminal with GPU access:

```bash
bash scripts/run_baselines.sh --config configs/baselines-reproducible.json --preflight
bash scripts/run_baselines.sh --config configs/baselines-reproducible.json --task-index 0 --device cuda
```

After checking that pilot, complete the remaining tasks and summarize:

```bash
bash -e <<'BASH'
for task in $(seq 1 26); do
  bash scripts/run_baselines.sh --config configs/baselines-reproducible.json --task-index "$task" --device cuda
done
bash scripts/run_baselines.sh --config configs/baselines-reproducible.json --summarize-only
BASH
```

Next, review the corrected cohort's validation learning curves and participant
variation. If the neural generalization gap persists, use small controlled
training/preprocessing investigations, with a fixed validation decision rule,
before a temporal or tensor extension. Keep the covariance comparator.

Cross-session evaluation is a subsequent separate study. All nine E recordings
are present, but no official E-label `.mat` files were found under the data
directory and the cross-session config still has `external_labels_dir: null`.
Obtain those labels, explicitly configure their directory, and keep E as test
only. The legacy `results/inm-v2` study is absent, so a tensor-effect claim or
stage-based tensor diagnosis is not yet supported by these baseline results.

No new real training was launched during this review.
