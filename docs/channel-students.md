# Reduced-channel supervised and distilled students

These studies extend the published-checkpoint ATCNet comparison. They do not
replace the existing covariance experiment or the separate EEGNet/Signal
Transformer study.

| Experiment | Training intervention | Inference |
|---|---|---|
| Existing covariance | Fit frozen/learned completion; keep published classifier frozen | Complete missing inputs, then classify |
| Q1 | Train a new ATCNet student from class labels | Classify observed signals directly |
| Q2 | Train the same student from labels and full-channel teacher predictions | Classify observed signals directly; teacher unnecessary |

Q1 asks how well direct reduced-channel training performs. Q2 asks whether
teacher guidance improves it, and whether a distilled student performs better
than learned covariance. None of these accuracies is assumed in advance.

## Independent selection and configuration

The existing experiment continues to use `python -m published_covariance` and
`configs/published-covariance.json`. Its code, configuration and launcher are
unchanged by the student extension.

The new runner requires `--experiment q1` or `--experiment q2`. Each has a
self-contained configuration, without inheritance from the covariance study:

- [Q1 configuration](../configs/atcnet-student-q1.json): distillation disabled;
  output `results/q1-atcnet-selected-channels-v1`.
- [Q2 configuration](../configs/atcnet-student-q2.json): distillation enabled;
  output `results/q2-atcnet-selected-channels-v1`.

No covariance fit or result is required to run either student study. They share
the hash-verified recording/checkpoint assets and pinned native model/data/mask
definitions. Each configuration pins those shared definitions: a change is
rejected until that experiment's contract is explicitly reviewed and repinned.
Covariance optimizer, completion, reporting and configuration changes do not
select or change the student experiment. Different source/settings/runtime
identities require a fresh output suffix; prior results are never overwritten.

## Available electrodes and model structure

The covariance protocol varies electrode identities between trials. To match
it, each participant/seed trains **two separate students**: one always receives
16 observed channels, the other always receives 6. Both preserve the 22 electrode
positions, use explicit availability masks and replace hidden inputs with zero
before encoding. No missing signal is reconstructed. A 22-slot input therefore
does not mean that the student receives 22 observed EEG signals.

Both students use the same native ATCNet graph as the teacher: EEG convolutions,
five temporal MHA/TCN branches and averaged classification logits. They have
115,172 total parameters, including normalization buffers. They start from
random initialization, without loading teacher weights. This experiment does
not measure the computational savings of a compact fixed-montage network.

The same participant/seed/channel-count initialization and batch-order streams
are used for Q1 and Q2. Masks use exactly the covariance experiment's namespace,
trial identities and subset-pool partitioning. Thus all three approaches can be
compared on identical E trials and observed electrodes. Training masks change
each epoch; validation and E each have five fixed repeats. Training, validation
and E electrode-subset pools are disjoint.

## Data, training and checkpoint selection

Use [BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/) with the
native structured T/E MAT recordings described in the
[published-checkpoint setup](published-covariance/README.md#dataset-and-launch).
Keep every subject and every artifact-flagged trial. Load the native 1,125-sample
crop without additional filtering. Each seed creates the same classwise T split
as covariance: **232 training and 56 validation trials**; all **288 E trials**
are reserved for evaluation after fitting finishes.

Student normalization is fitted on T training trials only, separately for each
seed, and remains fixed during validation and E evaluation. Q2 replays the
pretrained teacher's historical all-T scaler for teacher inputs only. The two
student studies therefore share preprocessing, while their normalization
contract differs from the published frozen classifier's historical contract.

Q1 and Q2 use the same Adam settings: learning rate 0.001, batch size 64, epsilon
1e-8, global gradient clipping at 1, at most 500 epochs, at least 100 epochs and
patience 100. Native ATCNet kernel regularization is retained in both. The
selected checkpoint maximizes validation balanced accuracy for that student's
channel count, then minimizes log loss, then retains the earlier epoch. Epoch
zero is eligible. E scores never select an epoch or distillation settings.

The objectives are

$$
L_{Q1}=\operatorname{CE}(y,z_S)+R(\theta_S),
$$

$$
L_{Q2}=\operatorname{CE}(y,z_S)+R(\theta_S)
 +\lambda\tau^2\operatorname{KL}
 \left(\operatorname{softmax}(z_T/\tau)
 \Vert\operatorname{softmax}(z_S/\tau)\right).
$$

Here the teacher sees the complete training recording and the student sees only
the masked recording. Q2 defaults to **lambda = 0.5** and **temperature = 4**;
these are declared starting settings, not accuracy-optimized values. Teacher
logits are computed only for T training trials and saved with their identities.
Teacher weights, dropout and BatchNorm stay frozen. Q1 rejects teacher targets
and has zero distillation weight. Validation uses labels, without teacher loss.

This follows standard teacher-probability distillation
([Hinton, Vinyals and Dean, 2015](https://arxiv.org/abs/1503.02531)). Distillation
from higher-density to lower-density EEG also has motor-imagery precedent
([Huang, Chen and Wei, 2022](https://arxiv.org/abs/2212.03329)); this experiment
does not claim to reproduce their similarity-keeping objective.

## Commands and smaller cohorts

Use the existing TensorFlow GPU environment from the ATCNet setup; no new
dependencies or package installation are required. Default plans declare nine
participants, three seeds, **54 student fits and 270 E prediction cells per
study**. Neither command runs the other experiment.

```bash
python -m channel_students plan --experiment q1
python -m channel_students plan --experiment q2

# A smaller independent plan: three participants, one seed, one student count.
python -m channel_students plan --experiment q1 \
  --subjects A03 A04 A09 --seeds 0 --retained-counts 16
```

For direct stage execution, keep the selector, cohort, configuration and output
identical for every command. The full Q1 workflow is:

```bash
output=results/q1-atcnet-selected-channels-v1
python -m channel_students admit --experiment q1 --output "$output"
for subject in A01 A02 A03 A04 A05 A06 A07 A08 A09; do
  for seed in 0 1 2; do
    python -m channel_students fit-task --experiment q1 --output "$output" \
      --subject "$subject" --seed "$seed"
  done
done
python -m channel_students seal-cohort --experiment q1 --output "$output"
for subject in A01 A02 A03 A04 A05 A06 A07 A08 A09; do
  for seed in 0 1 2; do
    python -m channel_students evaluate-task --experiment q1 --output "$output" \
      --subject "$subject" --seed "$seed"
  done
done
python -m channel_students report --experiment q1 --output "$output"
python -m channel_students.verify --output "$output"
```

For Q2, replace every `--experiment q1` with `--experiment q2` and select its
own `results/q2-*` output. `--config` supports an independent configuration for
either study. For a smaller run, repeat `--subjects`, `--seeds` and
`--retained-counts` consistently across all stages, with a fresh output. Shorter
training budgets belong in that study's separate configuration and are labeled
debugging runs, not full-protocol measurements.

GPU admission checks native checkpoint tensors/attention, hidden-input
invariance, zero gradients through hidden signals and an actual generated-input
student update. All fitting must finish and be sealed before any E signals or
labels are opened. Selected H5 weights replay T validation before sealing;
saved E predictions replay inference. Independent NumPy verification recomputes
metrics and aggregation from all saved predictions. Interrupted tasks are
preserved and prevent a complete report; failed participants are never dropped.

Each result directory contains `report.md`, `report.json`, `verification.json`,
`audit.json`, `identity.json`, GPU admission evidence, selected student H5
weights, histories, normalization, splits, training masks/order and per-trial E
probabilities/masks. Aggregate repeats, then seeds, then participants equally.

## Compare the three completed studies

This optional comparison reads completed reports; it does not fit any model or
rewrite any source study. It requires matching student configurations, initial
weights, overlapping batch/mask schedules, splits, recording hashes and E masks.

```bash
python -m channel_students.compare \
  --covariance results/published-covariance-validation-v1 \
  --q1 results/q1-atcnet-selected-channels-v1 \
  --q2 results/q2-atcnet-selected-channels-v1 \
  --output results/channel-comparison-v1
```

The comparison reports Q1 minus learned covariance, Q2 minus learned covariance,
and Q2 minus Q1, with participant-bootstrap intervals. It averages 16/6 effects
equally. The last comparison isolates teacher guidance within the matched
student recipe. Early stopping can produce different completed epoch counts,
so the common maximum budget and stopping rule are matched, not necessarily
the realized amount of training.

## Interpretation

These are exploratory studies. The published teacher's original checkpoint
selection using E is not conclusively excluded, and its training/scaler saw all
T trials, including the later student-validation subset. Q2 inherits that
historical teacher provenance. Q1 does not consume teacher supervision; prior
project exposure to E outcomes still precludes an untouched-test claim.

Students update their entire classifiers; covariance updates 253 completion
parameters while preserving the classifier. Their fitting costs and historical
training information differ. Report accuracy and fitting/inference costs with
those distinctions, and retain all declared outcomes. These comparisons test
alternative adaptation strategies rather than assuming one must win.
