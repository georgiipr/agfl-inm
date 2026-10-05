# Reconstruction and fixed-probe diagnostics

`audit_reconstruction(sources, ordered_probe, cfg)` uses the fitted
`features_ordered` probe, including its training-fitted scaler, for all three
representations: full validation features, normalized zero-fill, and frozen
Tucker completion. It does not fit a probe or factors. The probe must match the
task and the exact validation sample ID order.

The validation banks are generated from stable trial IDs with the existing
mask-bank protocol: full input once, then five repeats each for static and
dynamic random loss at 16 and 6 retained channels. One mask bank is shared by
all three representations in a condition/repeat. Spatial and 11-channel
conditions are rejected. For each degraded bank, the Tucker solver selects
available values before arithmetic and completion is checked to preserve every
observed feature exactly. Labels are read only after all banks have been
completed.

## Returned condition fields

Each row in `conditions` contains condition identity (`condition`, `pattern`,
`retained`, `repeat`), `mask_sha256`, `mask_shape`, and `hidden_entries`.
`mask_identities` maps all three scoring methods to that same digest.
`hidden_squared_error_sum` and `hidden_target_energy_sum` are pooled float64
sums over all hidden feature entries; `hidden_nrmse` is
`sqrt(hidden SSE / hidden target energy)`, matching the legacy study's
`_reconstruction_scores` calculation. This is not the average of per-trial
ratios. Full input has no hidden entries, so both NRMSE values are null. A
zero-energy degraded bank also has null NRMSE. Zero-fill has NRMSE one when
hidden target energy is nonzero.

`metrics` contains `balanced_accuracy`, `log_loss`, per-class recall, class
counts, and confusion matrix for `full_features`, `zero_fill`, and
`tucker_completion`. A missing class has null recall and count zero; balanced
accuracy follows the standard average over classes present in the labels.
`paired_tucker_minus_zero_fill` reports differences for balanced accuracy,
log-loss, and each class recall; unavailable recall differences are null.

`per_trial` is keyed by stable `sample_id` and records its label, hidden SSE,
hidden target energy, per-trial NRMSE, and true-label log-loss for each method.
`class_stratified` summarizes hidden SSE/energy and mean log-loss by true class
for post-fit description, including pooled class-specific NRMSE when energy is
nonzero. Empty groups have count zero and null summaries.
Class labels affect these metrics and group summaries only; they do not affect
mask generation or reconstruction.

Keep the five mask repeats separate in audit outputs. Aggregate repeats within
seed, then seeds within participant, then participants equally. This diagnostic
is exploratory on reused validation; neither an error/accuracy relation nor an
NRMSE value establishes causation, classification adequacy, or a reason to
change Tucker rank.
