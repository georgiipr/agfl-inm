# Experiment pipeline

The full preset is `configs/study.json`: four models, nine individually trained
participants, three seeds, **108 end-to-end training runs**. A task owns one
participant/seed and runs its selected models sequentially. `pilot.json` has one
seed (36 runs); `debug.json` has EEGNet on three participants, one seed and
12 epochs (3 runs). Presets
inherit the full configuration and use distinct output directories.
`smoke.json` runs all four model paths on one participant with the short debug
budget (4 runs), exercising both tensor paths before a full run.

The two EEGNet-only comparisons retain EEGNet and EEGNet with tensors:
`eegnet_3subjects.json` uses A03/A04/A09 (6 fits), and
`eegnet_all_subjects.json` uses all nine participants (18 fits). Both use seed 0,
batch size 64, weight decay 0.001 and the restored 500-epoch recipe. They have
separate output directories, calibration records, selected checkpoints and
reports. Each comparison trains
its own declared participants; the nine-participant report does not pool results
from a different experiment.

## Data and fitting

Load cue-aligned four-second T-session trials from BCI IV 2a, include marked
artifact trials as in the original baseline, and retain the canonical 22 EEG
channels at 250 Hz. Artifact counts remain recorded. Four one-second
availability windows describe masks and tensor axes. Full inputs use the
original 2–30 Hz zero-phase run-level bandpass before trial extraction,
including native GDF gaps and run boundaries. Each model receives the entire
trial, with ordered temporal information.

For each degraded trial, remove hidden raw intervals before filtering. Each
electrode is processed independently over its remaining contiguous observed
spans. Synthetic outages apply within the cue-aligned trial; outside-trial
recording context remains observed. Native gaps and run boundaries are retained.
Electrodes observed throughout the trial reuse the original filtered input.
Partial-electrode views are cached by trial, electrode and availability schedule
and reused across scenarios and models. This preserves the historical full-input
preprocessing without allowing hidden intervals into observed degraded inputs.

Persist a class-stratified 60/20/20 train/validation/test split per participant
and seed. Both backbones and their tensor variants use that same split. Fit
channel means and standard deviations only on training samples in float64,
then store and apply them in float32, with the original minimum standard
deviation of 1e-8. If tensor models are selected, fit one shared Tucker
channel/sample factor pair on
normalized full-channel training windows. Factors and normalization remain
fixed afterward. There is no supervised common-encoder pretraining or frozen
feature cache.

Train each model once, on all 22 channels. All backbone, MHA and classifier
parameters receive gradients. The production recipe restores 500 epochs, no
early stopping, batch size 64, AdamW learning rate 0.001, weight decay 0.001,
balanced training-class weights, cross entropy, ten warmup epochs followed by
cosine decay, and no gradient clipping. Debug and smoke deliberately use a
short budget. PyTorch DataLoader uses shuffle=True, a seed-initialized generator
that advances across epochs, zero workers and no dropped final batch.
Use one CPU thread and deterministic CUDA settings as in the original recipe.
Full-channel model calls bypass completion, signal selection and MHA key masks.

For class c, weights are N_train / (4 * N_train,c). Loss is the mean of these
weights times each example's cross entropy, matching the original implementation;
it is not normalized by the sum of weights. Validation uses those same
training-fitted weights and averages batch losses by example count. Evaluate
only 22-channel validation during training; select the epoch with the lowest
validation loss. Strict improvement is required; ties keep the earlier epoch.
Unweighted validation log loss remains descriptive. Test labels and
degraded validation scores never participate in checkpoint selection.

Fixed four-head MHA uses the original Linear(dim, 3*dim) Q/K/V and Linear(dim,
dim) output projections and explicit scaled softmax scores, with the original
availability mask excluding missing keys. Initialization uses a separate CPU
generator seeded by `(model_seed + 104729 * (layer_index + 1)) % 2**63`,
restoring the backbone RNG state after construction. EEGNet keeps the original
convolution/attention/classifier construction order. There is no attention
registry, factory or switch.

Save the selected checkpoint before evaluation. After selection, predict on
validation and then held-out test for 22, 16, 11 and 6 channels. Each degraded
count has random/spatial static and dynamic masks, five repeats by default.
There are 13 scenarios and 61 prediction conditions per partition, 122 metric
rows per full-preset model run. All sweeps use the same selected weights, with
no optimizer steps or factor updates. Masks are shared across all four models.

## Outputs and aggregation

Use `results/inm-v5` for full, separate pilot/debug directories for presets.
The EEGNet comparisons use `results/eegnet-a03-a04-a09-restored-v5` and
`results/eegnet-all9-restored-v5`. Older result directories remain intact.
Schema/source/environment identities prevent reusing previous-protocol results.
Model records live in `artifacts/Axx_seed_s/MODELS/<model>/`. Each contains
configuration, training history, selected `checkpoint.pt`, checksum/selection
metadata and all validation/test metrics. A participant/seed calibration cache
stores train-only normalization and factors. Splits, trial IDs, dataset source
checksums, exclusions, settings, source hashes and package versions accompany
the study. CUDA/cuDNN versions, GPU names and deterministic runtime settings are
also recorded. Each condition records a preprocessed-input hash; paired reports
require both input and mask equality. Do not edit a running study's inputs or source.

Rerunning a task skips complete provenance/checksum-verified runs. If training
completed but evaluation was interrupted, restore the selected checkpoint and
finish inference rather than train again. Failed models are recorded while the
remaining declared models continue; errors and unfavorable outcomes are kept.

Balanced accuracy is primary; ordinary accuracy and macro-F1 are also reported.
For each condition, average mask repeats, then seeds within each participant,
then equally over configured participants. Full means require all declared
participants/seeds; unfinished means stay blank. Completed individual runs remain
visible. Subject SD describes participant variation, not independent seeds.
Paired tensor gains require matching dataset, split, calibration and mask hashes.
Robustness gives equal weight to 22/16/11/6 counts within a loss pattern. Bootstrap
intervals resample participants after averaging paired seed differences; they are
pointwise exploratory intervals with no multiple-comparison correction.

Reports update after runs. Optional PNG/PDF plots of validation/test availability
and paired robustness are rebuilt from saved tables through `--summarize-only
--plots`. Plotting never runs checkpoints. See [README](../README.md) for commands.

## Research limits

This is a within-session offline availability study, not competition T-to-E
performance or unseen-participant generalization. Raw-window tensor completion
may fail to recover task information, especially under severe or grouped loss. Full-channel pass-through does not assert accuracy or tensor benefit;
these require experimental measurements. Hidden-sample NRMSE is secondary and is
scored only after fitting, never used as a model input or tuning objective.

The v5 recipe restores the original full-channel run filtering as well as
training and initialization. Degraded trials are refiltered after removing raw
outages; zeroing an already filtered intact dynamic trial would leak hidden
intervals into observed inputs. This protocol assumes known recording context
outside the evaluated trial and uses offline zero-phase filtering. Historical
accuracy replication must be measured with matching participants, seeds and
runtime versions; static source inspection cannot establish numerical parity.
