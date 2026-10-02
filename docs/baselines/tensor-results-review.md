# Completed tensor follow-up: decision and next work

The declared MHA study in `results/inm-tensor-followup-v1` is complete:
162/162 classifier fits across nine participants and three seeds. Its report
contains no invalid results, failures, identity conflicts, or pairing issues.
The current source, package versions, and configuration match its study identity.
All 27 calibration files were checked against their SHA-256 records; all 162
selected history rows agree with the saved selection metrics. Within each task,
the six heads share identical feature, calibration, and evaluation-mask hashes.

## Validation evidence

Balanced accuracy (%), averaging seeds within participant and then participants
equally. All participants have three seeds. The robustness selection score
equally weights full input and static/dynamic random losses at 16/6 channels,
after averaging mask repeats within each condition.

| Representation | Training | Full-input validation | Robust validation selection score |
|---|---|---:|---:|
| Baseline | Full | 40.62 | 36.48 |
| Tensor completion | Full | 41.27 | 36.97 |
| Tensor core | Full | 39.70 | 35.04 |
| Baseline | Mixed | 41.30 | 38.69 |
| Tensor completion | Mixed | 41.71 | 38.89 |
| Tensor core | Mixed | 40.29 | 36.35 |

Tensor core reduces the declared validation selection score by 1.44 percentage
points under full training and 2.34 under mixed training. Completion's gains
are only 0.50 and 0.20 points respectively. These are independently selected
validation maxima, so small differences are selection-biased diagnostics, not
independent proof of an effect. These results do not justify expanding the
tensor rank/ridge search or adopting tensor core as the default.

Shared encoder pretraining averages 44.07% selected full-input validation BA.
Its mean recorded training accuracy at those epochs is 45.66%. That training
metric is collected during optimization with dropout active and changing
weights, so it must not be treated as clean evaluation-mode training BA or
used alone to diagnose underfitting. The modest pretraining validation score
shows that the weakness precedes Tucker compression, although it does not
identify its cause.

## Test outcomes, separate from configuration decisions

Full-input test BA for full/mixed training is 30.57%/32.33% for baseline,
31.26%/31.11% for completion, and 29.35%/31.22% for core. Reported paired
core-minus-baseline robustness differences are negative for all eight
regime/pattern comparisons (approximately -1.31 to -0.61 points). Every
reported 95% subject-bootstrap interval spans zero. These exploratory,
uncorrected intervals do not establish a gain or prove equivalence.

The test outcomes are final evaluation evidence for this setting. Do not use
them to select new ranks, epochs, or architectures. The earlier spatial EEGNet
and covariance results use different feature/model pipelines and are not
matched controls for attributing a tensor effect.

## Reporting caveat

The inherited `report/summary.md` paragraph and `progress.json` field incorrectly
describe checkpoint selection as full-input-only. Actual classifier histories
confirm the declared five-condition robust validation selection. Shared encoder
pretraining still uses full-input validation. This review records the correction;
the original experiment artifacts and source are preserved. Correct the reporter
in a separately tested maintenance change before any new study, using a fresh
output identity if source changes. No retraining is needed to interpret this
completed study with the correction.

## Next step

No additional GPU run is required to finish the current 12-session plan and
its declared six-arm tensor comparison. The optional ten-arm control study and
candidate rank/ridge runs have not been performed. Leave them unlaunched for
now: there is no established tensor gain that requires further attribution.
This is a conclusion about the tested configuration, not all tensor methods.

For the original accuracy objective, return to the reproducible spatial EEGNet
baseline with the flatten head and full-input training. The completed temporal
comparison favored flatten for that regime using its predefined validation
rule (60.86% versus 57.08%); see
[temporal-results-review.md](temporal-results-review.md). Keep covariance as a
simple reference. This recommendation is for full-input accuracy and does not
assert that full training is preferable for electrode-loss robustness.

The next bounded implementation task should audit the simpler baseline's
training and generalization before adding architecture:

1. Re-evaluate its saved selected checkpoints on clean training and validation
   data; inspect per-class behavior and participant variation. Use its saved
   normalization, splits, channel order, and provenance.
2. Check trial/cue alignment and preprocessing against the declared protocol,
   and separate optimization behavior from the training/validation gap already
   observed in [corrected-results-review.md](corrected-results-review.md).
3. Only if the audit supports a specific intervention, predeclare a small
   validation-only training ablation with matched data, seeds, and a fresh
   output directory. Do not include test scores in that choice.

The current within-T-session protocol does not establish SOTA or cross-session
performance. A later benchmark study must declare the matching evaluation
protocol and supply any required official E-session labels before execution.

This review required no new training, package installation, source edits, or
changes to completed results.
