# Declared experiment and reporting

The current schema-2 study reduces the attention matrix to MHA and Performer.
Its output is `results/inm-v2`; previous studies retain their own configuration
and results. The participant cohort, three seeds, EEGNet-derived encoder,
splits, calibration and evaluation-mask design are unchanged. Signal Transformer
remains available in the model library and is outside this experiment matrix.

## Cohort and splits

Nine independently trained BCI Competition IV 2a participants, A01–A09; seeds
0, 1, 2. Each participant's T session is stratified by class into approximately
60/20/20 train/validation/test trials. Fractions are rounded separately within
each class. This is **within-session, subject-dependent** evaluation, not the
official train-on-T/test-on-E competition protocol. E files are not required.
No participant pooling or participant exclusion is based on measured accuracy.

Cue classes 769–772 map to left hand, right hand, feet and tongue. The first 22
EEG channels are retained, excluding EOG. Trials begin at the cue and use 1,000
samples at 250 Hz. Artifact-flagged, boundary-crossing and unusable/nonfinite
trials are excluded consistently for every arm. Saved data metadata includes
source checksums, excluded counts and remaining class counts.

Each 250-sample window is independently filtered with the existing fourth-order
2–30 Hz Butterworth SOS zero-phase routine. No filter padding, convolution or
per-trial normalization may bring a hidden window into an observed one. This
offline processing is not causal and the short-window boundaries can affect
accuracy; it is chosen to define a valid dynamic-availability simulation.
Raw-channel and learned-feature means/stds use **only the training partition**.

## Shared calibration and classifier matrix

For each subject/seed, a single full-input MHA-supervised encoder is fitted and
selected on full-input validation. Its frozen features are shared across all
arms. Tucker ranks are `Rc=4`, `Rf=4`, ridge `0.001`; 30 projected alternating
fit epochs on full **training** features. There is no test-based rank search.
Both regimes therefore have full-channel calibration; only downstream
classifier training has differing availability.

Each of the two attentions, MHA and Performer, has four main fits:

| Representation | Full classifier training | Mixed classifier training |
|---|---|---|
| Baseline | Yes | Yes |
| Tucker core | Yes | Yes |

MHA additionally has tensor completion, separable linear core and training-mean
completion under both regimes: six controls. Total `(2 × 2 + 3) × 2 = 14` arms, or 378 classifier fits across 27 participant/seed tasks.
There is no no-attention option. Model and attention choices remain separate.

The shared feature pretraining may favor MHA; disclose that choice. The linear
control has two trainable maps with `G = Wc X Wfᵀ`. It controls supervised
separable compression, not unrestricted dense compression. Its shapes match
the Tucker core but its objective and trainable parameter count differ. Baseline
missing placeholders and completion pooling also differ as documented in
`model_adaptation.md`; controls are needed before attributing a gain solely to
multilinear structure.

Attention uses dimension 32 and four heads. MHA uses standard scaled dot-product
attention; Performer uses 64 fixed random features per head. Both are applied
spatially within each window. Neither uses temporal attention or an output gate.
This is a method comparison, not a claim that the approximation is faster or
more accurate for these short token sequences.

All supervised fits use the declared shared CE/AdamW settings, maximum 250
epochs, batch 32, initial target LR 0.001 with 10-epoch warmup/cosine scheduling,
weight decay 0.0001, gradient clip 1, and early stopping after at least 75 epochs
with patience 50. The highest **full-input validation balanced accuracy** wins;
unweighted validation log loss breaks ties. Test scores never select an epoch.
There is no augmentation unrelated to the declared availability masks and no
arm-specific tuning. Deterministic PyTorch mode is requested; unsupported
operations fail visibly instead of silently weakening that setting.

## Availability and evaluation

Full training observes all 22 channels. Mixed training samples 22, 16 or 6
channels per trial/epoch; degraded samples use random static or random dynamic
masks. Neither 11-channel inputs nor spatial-loss sampling is used for classifier
training. Every model sees the same keyed training masks, trial order, splits
and evaluation masks for its participant/seed/regime.

Each selected head receives the full scenario and four patterns at 16, 11 and
6 retained channels on validation and test. Patterns are static random, static
spatial, dynamic random and dynamic spatial. Dynamic masks use an A–B–A schedule
with exact retained counts in every window. There are five deterministic repeats
per degraded condition; full input is scored once. Nonfull channel combinations
are partition-disjoint; the full mask is the unavoidable shared exception.

Thus each completed fit contains `2 × (1 + 12 × 5) = 122` metric rows. It is
trained once and evaluated repeatedly, not trained 122 times. The full-only
regime directly tests train/evaluation channel-count mismatch. Degraded validation
scores are descriptive outputs after checkpoint selection, not extra selection
criteria. After examining held-out test results, a later modified study should
be described as exploratory rather than an untouched confirmatory evaluation.

## Scores and uncertainty

For four classes, balanced accuracy is

\[
\operatorname{BA}=\frac14\sum_{k=0}^{3}
 \frac{\#\{i:y_i=k,\widehat y_i=k\}}{\#\{i:y_i=k\}}.
\]

Primary: balanced accuracy. Secondary: accuracy, macro-F1, full-input degradation
and hidden-feature reconstruction NRMSE. Raw records also preserve class supports,
per-class metrics, confusion matrices and ROC-AUC when defined. Classifier
parameter counts, selected epoch, fit duration, exact mask hashes and source
identities are retained. Runtime metadata is descriptive, not a controlled
attention-speed benchmark.

First average mask repeats, then seeds within each participant, then the nine
participant means equally. Do not pool trials across participants or treat masks
and seeds as independent participants. Report subject SD separately from a
confidence interval. Nine-subject means are left blank until all required
participant/seed fits for that arm are complete.

For each attention/regime, compute paired Tucker-core-minus-baseline differences
with the same masks. Average seed differences within a participant, then bootstrap
those participant means 2,000 times to obtain pointwise 95% percentile intervals.
These intervals are exploratory, without multiple-comparison correction; the
MHA secondary controls have score tables but no separate control-specific paired
intervals in this first version.

For each of the four outage patterns, robustness is the equally weighted mean
of the full, 16-, 11- and 6-channel scores. It is a four-condition average, not
a trapezoidal area under a continuous loss curve. Positive paired gains favor
the tensor route; positive degradation is a drop from full-input performance.

Reconstruction NRMSE is computed only over hidden, standardized feature entries:

\[
\operatorname{NRMSE}=\sqrt{
\frac{\sum_{M=0}(\widehat X-X)^2}{\sum_{M=0}X^2}}.
\]

This is a dimensionless relative feature error, not raw-waveform recovery.
Full-input NRMSE is undefined because nothing is hidden. A zero reference energy
also yields a blank value. Reconstruction scores never fit factors or select
classifiers. Lower reconstruction error need not improve classification.

## Output layout

```text
results/inm-v2/
  study.json                         exact config/source/environment identity
  datasets.json                      one checked data identity per participant
  report/                            download this folder
    summary.md                       readable progress and main comparisons
    accuracy_by_run.csv               completed fits, useful while jobs run
    accuracy_by_subject.csv           seed means per participant/condition
    accuracy_table.csv                complete all-participant means and SD
    robustness_table.csv              equal-weight availability averages
    paired_tensor_gain.csv            paired gains and subject intervals
    progress.json                     missing fits, errors and invalid records
    study_manifest.json              full settings, hashes, protocol
    datasets.json                     remaining trials and exclusions
    plots/                           optional PNG/PDF figures
    plots_status.json                optional figure-completeness manifest
  artifacts/                         keep on cluster for restart/audit
    A01_seed_0/
      dataset.json
      split.json
      splits/
      calibration.pt                 frozen encoder/features/stats/factors
      calibration.json               checksum, calibration/selection metadata
      encoder_history.json
      factor_history.json
      task_status.json
      ARMS/<attention>__<route>__<regime>/
        config.json
        history.json
        result.json                  only published after all evaluations
        error.json                   only if a fit failed
```

Selected classifier weights remain in memory through evaluation and are not
saved as separate checkpoints. Shared calibration state is saved for deterministic
restarts. An interrupted classifier fit restarts from its seeded initialization.
Complete matching results are reused; config/source/data/split changes cannot
silently reuse an old score. Concurrent array tasks use file locks and atomic
publication; reports reject incomplete rows or inconsistent paired provenance.

## Interpreting the result

Read the full-input table first, then availability curves and paired gains. A
beneficial robustness tradeoff may coexist with a full-input accuracy cost.
If Tucker helps only a subset of attentions/participants, say so and retain the
complete cohort in the primary table. Compare MHA controls to distinguish masked
inference, completion and generic compression explanations. A null or negative
result is evidence about the chosen ranks, features and protocol, not a reason
to omit participants or change the held-out scoring rule.
