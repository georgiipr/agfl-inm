# Experiment pipeline

The full preset is `configs/study.json`: four models, nine individually trained
participants, three seeds, **108 end-to-end training runs**. A task owns one
participant/seed and runs its selected models sequentially. `pilot.json` has one seed (36 runs); `debug.json`
has EEGNet on three participants, one seed and 12 epochs (3 runs). Presets
inherit the full configuration and use distinct output directories.
`smoke.json` runs all four model paths on one participant with the short debug
budget (4 runs), exercising both tensor paths before a full run.

## Data and fitting

Load cue-aligned four-second T-session trials from BCI IV 2a, exclude marked
artifacts, and retain the canonical 22 EEG channels at 250 Hz. Four one-second
filter windows are processed independently with the 2–30 Hz zero-phase bandpass.
No filter crosses into a window that may later become unavailable. Each model
still receives the entire trial, with ordered temporal information.

Persist a class-stratified 60/20/20 train/validation/test split per participant
and seed. Both backbones and their tensor variants use that same split. Fit
channel means and standard deviations only on training samples. If tensor
models are selected, fit one shared Tucker channel/sample factor pair on
normalized full-channel training windows. Factors and normalization remain
fixed afterward. There is no supervised common-encoder pretraining or frozen
feature cache.

Train each model once, on all 22 channels. All backbone, MHA and classifier
parameters receive gradients. Use the declared AdamW, loss, warmup/cosine
schedule, clipping and early-stop budget. Evaluate only 22-channel validation
during training; select the epoch with highest validation balanced accuracy,
breaking ties with lowest unweighted validation log loss. Test labels and
degraded validation scores never participate in checkpoint selection.

Save the selected checkpoint before evaluation. After selection, predict on
validation and then held-out test for 22, 16, 11 and 6 channels. Each degraded
count has random/spatial static and dynamic masks, five repeats by default.
There are 13 scenarios and 61 prediction conditions per partition, 122 metric
rows per full-preset model run. All sweeps use the same selected weights, with
no optimizer steps or factor updates. Masks are shared across all four models.

## Outputs and aggregation

Use `results/inm-v3` for full, separate pilot/debug directories for presets.
Schema/source/environment identities prevent reusing previous-protocol results.
Model records live in `artifacts/Axx_seed_s/MODELS/<model>/`. Each contains
configuration, training history, selected `checkpoint.pt`, checksum/selection
metadata and all validation/test metrics. A participant/seed calibration cache
stores train-only normalization and factors. Splits, trial IDs, dataset source
checksums, exclusions, settings, source hashes and package versions accompany
the study. Do not edit a running study's inputs or source.

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
