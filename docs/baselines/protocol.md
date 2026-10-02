# Baseline protocol implementation

The `agfl-baselines-v1` configs declare exactly these arms in this order:

1. `eegnet_reference/full` — neural, full and 12 degraded evaluation conditions.
2. `masked_eegnet/full` — neural, full and 12 degraded evaluation conditions.
3. `masked_eegnet/mixed` — neural, full and 12 degraded evaluation conditions.
4. `covariance/full` — classical, full-input only; available alongside the neural arms.

Each has one fit for each of nine participants and three seeds (108 fits).
The default `arms` declaration remains this complete comparison matrix. For
execution before the classical implementation is available, an optional
top-level `execution_arms` array may name a non-empty subset such as
`["eegnet_reference__full", "masked_eegnet__full",
"masked_eegnet__mixed"]`. It limits executed fits without changing the
declared comparison matrix; the session-07 covariance implementation is now
available as `covariance__full`.
Neural evaluation includes random-static, spatial-static, dynamic-random, and
dynamic-spatial loss at each of 16, 11, and 6 retained electrodes. The protocol
API names full coverage `full`; the legacy mask scenario calls it `full_22`.
The five validation banks used for robust epoch selection are a separate subset;
11-channel and spatial-loss evaluation must never enter epoch selection.
`within_session` stratifies T-session data 60/20/20 and is the initial default.
`cross_session` stratifies T-session training/validation 80/20 and uses E only
for test; E labels must be supplied explicitly through the loader contract.
Both use window-local 2–30 Hz filtering, with four 250-sample windows at 250 Hz.
The declared EEGNet defaults are temporal kernel 64, F1=8, D=2, F2=16,
spatial kernel [22,1], pooling [4,8], separable kernel 16, and dropout 0.5.
`inm.baselines.eegnet.EEGNetClassifier` consumes raw trials as
`[B,22,4,250]`; it selects hidden channel/window samples with `torch.where`,
then applies temporal convolution, depthwise spatial filtering over all 22
electrodes, ELU, average pooling by 4, separable temporal convolution, ELU,
and average pooling by 8. Its `forward_features` method returns the retained
sequence `[B,T,F2]` for optional temporal ablations. `mask_conditioned=True`
adds the original flattened 88 mask flags immediately before classification.
Both modes return logits. Constructor settings are available through
`constructor_settings()` and the model applies EEGNet max-norm constraints
through `clip_weights()`.

The cited ARL reference implementation uses this same temporal → depthwise
`[Chans,1]` spatial → ELU/pool → separable temporal block order, with F1=8,
D=2, F2=16, kernel length 64, pooling 4 then 8, dropout 0.5, depthwise
max-norm 1, and classifier max-norm 0.25. The baseline implementation adapts
it to segmented availability masks, optional mask conditioning, PyTorch
BatchNorm defaults, and logits (the Keras reference ends in softmax). It uses
fixed 22-channel, four-window, 250-sample defaults; it does not claim identical
cross-framework initialization or BatchNorm numerics.

Training defaults are 250 epochs, batch 32, AdamW LR 0.001, weight decay
0.0001, 10 warmup epochs, gradient clip 1, minimum 75 epochs, patience 50.
Selection defaults to full-input validation balanced accuracy with validation
log-loss tie-break. The robust policy averages balanced accuracy equally across
full, random-static 16, dynamic-random 16, random-static 6, and dynamic-random
6 validation banks, with mean validation log-loss as tie-break. Test, 11-channel,
and spatial-loss scores are excluded from both policies. Configs declare five
repeats per degraded mask bank and 2,000 participant bootstrap repeats for future
paired intervals.

Relative paths resolve from each config file's directory. In this checkout,
`configs/baselines.json` resolves `../../ml` to the sibling `ml/` directory and
`../results/baselines-v1` to `results/baselines-v1/`. The cross-session config
uses that same data directory and a separate `results/baselines-cross-session-v1/`
output directory.

`python -m inm.baselines --config configs/baselines.json --smoke` writes to a
source-identity-suffixed sibling of `results/baselines-v1/` (for example,
`results/baselines-v1-synthetic-smoke-<source-hash>/`). It runs one
`eegnet_reference__full` arm by default with generated synthetic data, records
`synthetic: true`, and emits no real-study task records. Its task result uses
`agfl-baseline-task-v1`; arm results use `agfl-baseline-task-result-v1` and
include validation/test metrics, exact availability-mask hashes, selected
epoch, parameter count, timings, checkpoint/history checksums, and provenance.
The session-07 covariance arm writes `model.npz` and contributes full-input
metric rows; neural-only `execution_arms` subsets remain useful for controlled
partial runs.
# Reporting handoff

`python -m inm.baselines --config configs/baselines.json --summarize-only`
calls `inm.baselines.reporting.summarize(cfg)` and reads per-arm
`artifacts/Axx_seed_n/ARMS/<arm>/result.json` and per-task `dataset.json` files. It does not import the
training/model modules. Outputs are written under
`<output_dir>/report/summary.md`: `accuracy_by_run.csv`,
`accuracy_by_subject.csv`, `accuracy_table.csv`, `degraded_table.csv`,
`full_input_degradation.csv`, `paired_neural_differences.csv`, plus
`pairing_issues.csv`, `invalid_results.csv`, `missing_results.csv`, and
`progress.json`.

Only `agfl-baseline-task-result-v1` records marked `complete` and
`synthetic: false` enter real reports. Each neural task requires one full row
and every configured repeat for all 12 degraded scenarios in validation and
test; covariance requires full-input rows only. Result identity includes
config/source hashes, protocol, data fingerprint, split ID, and synthetic
status. Pairing checks subject/seed, data/split identity, protocol, selection
policy, and every per-cell mask hash. Per-run values average repeats;
participant values require all configured seeds; cohort values require every
participant and are weighted equally. A corrupt or partial record is listed as
invalid and cannot complete a cell. Synthetic smoke output must be summarized
with a separate synthetic-aware report in a future implementation; this
real-report path rejects synthetic records.

`dataset_fingerprint` identifies the loaded signals before splitting and must
match across seeds for a participant. `data_fingerprint` additionally includes
the split, seed, and training-only normalization: it normally differs across
seeds. Reporting recomputes that prepared-data hash from each task's persisted
metadata, compares it to the result, then checks the underlying dataset identity
across seeds. It never rewrites the saved results to make hashes match.

Neural construction now seeds all RNGs before creating each model, as well as
before training. Previously the seed was set only inside `fit_classifier`, after
initialization; weights depended on ambient RNG state and arm execution order.
The completed original study remains a descriptive record of those executions.
Use `configs/baselines-reproducible.json` and its separate output directory for
the corrected replication. Do not merge these studies or resume new fits in the
original directory after source changes.

For a later real pilot review, start at
`<output_dir>/report/summary.md`, inspect `progress.json` and
`invalid_results.csv`, then check `accuracy_by_run.csv` against
`artifacts/A01_seed_0/ARMS/<arm>/history.json`, `result.json`, and neural
`checkpoint.pt` (or covariance `model.npz`). Session gates must see a complete
real A01/seed-0 record and these readable reports; cohort summaries remain
pending until all configured tasks finish.
