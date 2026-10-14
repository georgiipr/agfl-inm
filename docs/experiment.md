# Experiment pipeline

Schema v6 retains the two fixed-MHA backbones, each alone, with frozen covariance,
and with learned covariance. The full preset has nine participants, three seeds
and six variants: **162 variant evaluations**, **54 backbone fits** and **54
learned covariance calibrations**. A task owns one participant/seed and processes
its variants sequentially.

| Next EEGNet preset | Participants | Seed | Backbone fits | Learned fits | Variant evaluations |
|---|---|---:|---:|---:|---:|
| `eegnet_3subjects.json` | A03/A04/A09 | 0 | 3 | 3 | 9 |
| `eegnet_all_subjects.json` | All nine | 0 | 9 | 9 | 27 |

These independent experiments use separate output directories and reports.
The all-nine experiment trains all nine again; it does not pool pilot artifacts.
Debug uses one baseline on three participants. Smoke exercises all six paths
on one participant with 12 backbone epochs, three calibration epochs and one
mask repeat; its scores are only for workflow debugging.

## Data and backbone fitting

Keep the restored v5 design: cue-aligned four-second BCI IV 2a T-session trials,
all 22 EEG electrodes, artifacts included, 2–30 Hz zero-phase run filtering
before cutting trials, preserved native gaps/run boundaries, and stratified
60/20/20 training/validation/test trial splits. Fit channel mean/std on training
samples in float64; store/apply in float32 with the original 1e-8 std floor.

For outages remove hidden raw intervals before filtering each observed electrode
span. Outside-trial recording context remains observed. Uninterrupted electrodes
reuse original filtered values. The same observed-input cache serves all variants
and learned completion's masked training inputs.

Train each backbone end to end once on 22 channels: 500 epochs, batch size 64,
AdamW learning rate/weight decay 0.001, ten warmup epochs then cosine decay,
balanced class weights, no early stopping or gradient clipping. Keep original
seeded DataLoader order, initialization, one CPU thread and deterministic CUDA.
Class weights are `N_train/(4 N_train,c)`; average weighted example losses rather
than normalizing by their weight sum. Full-channel validation uses these same
training weights. Strictly lowest validation loss selects the epoch; exact ties
keep the earlier epoch. Test and degraded validation never select it.

Keep both architectures and their original MHA initialization streams unchanged.
Every completion variant reuses the same selected backbone checkpoint, recorded
by checksum. Full inputs bypass completion and MHA key masks exactly.

## Covariance calibration

Fit one shared 22×22 second moment on normalized full-channel training samples.
Frozen and learned arms initialize identical float64 SPD conditional-completion
operators. Frozen covariance has no supervised fitting. See
[covariance mathematics](covariance_math.md) for the solve and parameterization.

Learned covariance optimizes only 253 Cholesky coordinates through the frozen
selected backbone. Keep BatchNorm and dropout in evaluation mode. Sample
22/16/11/6 channels and random/spatial static/dynamic patterns from the reserved
train subset bucket, disjoint from validation/test subsets. Apply outages before
filtering. Combine the existing training-weighted classification loss, 0.1
hidden-training MSE and 0.0001 initialization anchor.

Use Adam at 0.001, batch size 32, clipping at 1, no weight decay/scheduler, and
100 fixed epochs. Full batches skip updates. Retain the final state and save
all training masks/order and history. No validation/test arrays enter this fit.
Runtime guards require unchanged backbone tensors and unchanged full-input
training-probe logits.

This extra supervised calibration is the learned arm's intervention. Frozen
covariance retains full-only fitting. The final-epoch rule differs from the
research branch's degraded-validation selection. This is a new matched study
in our restored setup, not reproduction of that branch's numerical results.
The learned arm evaluates unseen trials/subsets, not unseen outage families.

## Evaluation and aggregation

Freeze all weights before validation and held-out test evaluation. Keep the
previous 22/16/11/6 counts, four outage patterns, ABA schedule and five repeats:
61 conditions per partition, 122 metric rows per production variant. All models
receive identical observed-input and mask hashes. No sweep performs fitting.

Keep all participants/seeds and unfavorable outcomes. Average repeats, then
seeds within participant, then participants equally. Balanced accuracy is primary;
report accuracy, macro-F1, degradation and post-fit hidden-sample NRMSE too.
Better reconstruction does not guarantee better classification. Incomplete
means remain blank while completed individual runs stay visible. Robustness
averages 22/16/11/6 equally within each pattern.

Pair frozen-minus-baseline, learned-minus-baseline and learned-minus-frozen.
Require matching data/split/calibration, backbone checksum, masks and inputs.
Conflicting full-input scores exclude the pair. Pointwise bootstrap intervals
resample participant mean differences without multiple-comparison correction.

## Artifacts and resuming

Use fresh `results/eegnet-a03-a04-a09-covariance-v6` and
`results/eegnet-all9-covariance-v6` directories. Historical restored-v5 results
remain intact. Source/config/environment identity prevents previous-protocol
artifacts from being silently reused.

Each participant/seed directory preserves splits, sample IDs, dataset provenance,
normalization and covariance initialization. `BACKBONES/<backbone>/` stores its
shared selected checkpoint/history. `MODELS/<model>/` stores complete variant
weights, backbone checksum, selection record and all evaluation scores. Learned
variants retain `completion_history.json`, `completion_fit.json` and
`completion_training.npz`, with checksums. The study manifest records source,
configuration inheritance, launcher and runtime provenance.

Completed evaluation resumes from the saved variant checkpoint. Interrupted
calibration restarts deterministically from train-only initialization, reusing
the verified backbone within that study. Failed variants are recorded while
remaining variants continue; any failure makes the task exit unsuccessfully.
Changed source/settings need a new output directory.

The supplied launcher runs the CUDA preflight in `tests/test_covariance.py`
before first-stage training. It checks conditional estimates, hidden-value
isolation, full bypass, train-reserved masks, covariance gradients and frozen
backbone invariance. These checks have not been executed locally.

Reports update under a process lock. CSV/Markdown and optional PNG/PDF
availability/paired-gain plots stay separate per preset. Plotting reads saved
tables and never runs training. See the [README](../README.md) for commands.

## Research limits

This is offline within-session evaluation with recording context, not causal
streaming, unseen-participant generalization or competition T-to-E performance.
Covariance accuracy and historical full-input parity require cluster measurements;
static inspection cannot establish them. No covariance result is claimed yet.
