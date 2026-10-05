# Encoder investigation contract

## Question and boundaries

Determine whether the existing channel/window-local encoder retains class
information, whether training/generalization limits performance before Tucker,
and whether completion preserves discriminative information. Separate those
questions from reconstruction error. Implement a new read-only-input analysis
package at `inm/encoder_audit/`; do not alter the studied models or retrain them.

Primary inputs are `results/inm-tensor-followup-v1` (legacy frozen features,
27 calibrations and 162 heads) and `results/baselines-reproducible-v1`
(spatial EEGNet/covariance, 108 fits). Verify their actual coverage and source
identities; directory names are not proof. `results/inm-v2` is a single legacy
pilot and cannot replace the cohort. Existing reviews explain context, not
authoritative checksums. Earlier reported NRMSE about 0.56–0.69 is hidden-feature
error, not reconstruction accuracy or raw EEG recovery.

No new architecture, factor rank/ridge, filtering band, cue offset, loss, or
training budget is selected automatically. No cross-session transfer or George's
rewrite is imported. Existing test results are already known and must not be
reused for tuning. This audit uses training and validation only; its findings
are exploratory on reused validation, not independent confirmation.

## Immutable sources and trustworthy replay

Read original manifests, split records, checkpoints, histories, and calibration
files without invoking their writers or calling study initialization/resume.
Never edit a manifest to match current code. The new package changes the current
source digest; give the audit its own source/config/package identity and output
directory. Retain the original identities of its inputs separately.

Verify provided checksums before deserialization. Load trusted local tensor
checkpoints with `weights_only=True` where supported. A legacy payload requiring
another loader must be explicitly identified and justified for these trusted
local files; never fall back silently. Check imported replay source files against
the historical source map, not the entire now-expanded checkout. A mismatch
blocks faithful replay: report the exact paths; do not silently substitute code.

The legacy calibration stores encoder weights, factors, normalized feature
caches, and selection summaries. It does not store the whole pretraining MHA
head or the selected downstream heads. Inventory this explicitly. Reconstruct
the encoder only, and fit separately identified new probes. Do not claim to
reproduce original encoder-plus-head predictions from an absent head. Baseline
EEGNet selected checkpoints are reloadable and should be audited as such.

Restrict public analysis objects to train/validation arrays, IDs, and labels.
Legacy cache files may serialize all partitions together: loading that container
is permitted solely to extract train/validation immediately. Never score, fit,
summarize, or expose test rows. The same applies to mixed metric JSON records:
select allowed validation conditions before exposing them to analysis. Tests
must poison test labels/metrics/features and prove outputs remain unchanged.

## Data and normalization audit

Replay cue IDs, label order, channel order, source checksums, trial exclusions,
and the persisted splits. Do not generate a new random split. Validate 22 EEG
channels, four 250-sample windows, 250 Hz, cue offset zero, and the original
window-local 2–30 Hz preprocessing. Data preparation must not write into original
`.splits` paths. Keep baseline and legacy normalization conventions distinct:
their epsilon floors and saved feature statistics differ.

Using training rows alone, independently recompute and compare saved channel
statistics. Apply saved statistics to validation, replay frozen features, and
compare them with the saved cache. Encode both a complete batch and separate
batches in evaluation mode; verify BatchNorm/dropout remain frozen. Demonstrate
locality and hidden-NaN invariance. Default replay tolerances: float32 features
and probabilities `rtol=1e-4, atol=1e-5`; normalization statistics
`rtol=1e-6, atol=1e-8`. Record maximum errors. Tolerance changes require explicit
justification and a fresh protocol identity, not relaxation until tests pass.

Pair cross-study descriptive comparisons only after checking retained IDs,
labels, splits, preprocessing, and channel order. A failed cross-study pairing
does not erase valid within-study findings; mark the cross-study contrast
unavailable. Distinct model pipelines are never matched controls for a Tucker
effect, even if their trials match.

## Diagnostic models and fixed budgets

Re-evaluate the three saved neural baseline arms in evaluation mode, separately
on clean training and validation. Report BA, accuracy, log-loss, macro-F1,
per-class recall, confusion matrices, and gaps. Compare selected-epoch history
to replay. Training-time dropout/masked optimization accuracy is a separate
quantity. Keep the covariance result as context; do not refit it here.

For each legacy participant/seed, fit exactly three full-input probes:

| Key | Feature vector | Question |
|---|---|---|
| `features_ordered` | Flatten `[22,4,32]` to 2816 | Is class information linearly accessible in frozen features? |
| `features_window_mean` | Mean over windows, flatten `[22,32]` to 704 | How does removing window distinctions affect this probe? |
| `core_ordered` | Infer frozen Tucker core, flatten `[4,4,4]` to 64 | How does the existing compression affect this probe? |

Use a training-fitted `StandardScaler` per view and multinomial logistic
regression with L2, C=1, L-BFGS, max_iter=1000, tol=1e-4, random_state=0, and no
class weights or hyperparameter search. Use the installed sklearn API as in the
existing covariance classifier; do not depend on a removed `multi_class` option.
Nonconvergence is a recorded failure, not a silently accepted fit. Save numeric
coefficients/scaler/classes and prove probability equality after reload.

Add one `features_ordered_shuffled` negative control per task: permute training
labels with seed `task_seed + 700001`, fit the same fixed probe, evaluate against
true train/validation labels. Never permute validation or test labels. This makes
4 probes × 27 tasks = 108 small CPU fits, zero neural refits. A single shuffle
is a sanity control, not a significance test. Do not demand exactly 25% from a
finite real sample. Probe comparisons differ in dimensionality and regularization
geometry; they cannot isolate temporal order or prove all nonlinear information
was destroyed. Weak linear probes warrant a further controlled test, not a verdict.

## Reconstruction and class information

Use the same fitted `features_ordered` probe (its scaler included) to score full
features, normalized zero-fill, and Tucker completion under paired masks.
Never fit another probe on completed validation data. Observed features remain
exactly unchanged; factors are frozen, and labels cannot enter completion.
Allowed validation conditions are full and static/dynamic random loss at 16/6
electrodes, five repeats per degraded condition. No 11-channel or spatial-loss
validation conditions in diagnosis/selection, and no test output.

Compute hidden-entry NRMSE from sums of squared errors/targets per condition;
NRMSE=1 is normalized zero prediction. Full input has no hidden entries, so its
NRMSE is null, not zero. Also compute BA, log-loss, per-class recall, and paired
completion-minus-zero classification differences. Stratify reconstruction error
by true class for post-fit diagnostics only. A descriptive relationship between
trial error and log-loss does not establish causation or justify a rank search.
Report small denominators and empty groups explicitly. Do not transform NRMSE
into an invented accuracy percentage.

## Package interfaces and configuration

Use these interfaces, with primitive metadata and small dataclasses as needed:

| Module | Public entry points |
|---|---|
| `protocol.py` | `load_config(path)`, `tasks(cfg)`, `audit_identity(cfg)` |
| `inventory.py` | `inspect_inputs(cfg)` (stdlib, no checkpoint deserialization) |
| `artifacts.py` | `load_task_sources(cfg, subject, seed)` with train/validation-only views |
| `alignment.py` | `audit_alignment(sources, cfg)` |
| `checkpoints.py` | `audit_checkpoints(sources, aligned, cfg, device='cpu')` |
| `probes.py` | `fit_probes(sources, cfg, directory)`, `load_probe(path)` |
| `reconstruction.py` | `audit_reconstruction(sources, ordered_probe, cfg)` |
| `study.py` | `run_task(cfg, task_index, device='cpu')`, `run_smoke(cfg, output_dir)` |
| `reporting.py` | `summarize(cfg)` returning paths and complete/partial status |

Create `configs/encoder-audit.json`, schema `agfl-encoder-audit-v1`, subjects 1–9,
seeds 0–2. Config-relative paths: `data_dir: ../../ml`,
`baseline_dir: ../results/baselines-reproducible-v1`,
`legacy_dir: ../results/inm-tensor-followup-v1`,
`output_dir: ../results/encoder-audit-v1`. Include the fixed budgets, conditions,
and tolerances above. Reject unknown selection/search parameters, test partitions,
cross-session input, invalid masks, duplicate subjects/seeds, and any output path
equal to, inside, or containing either input study or the data directory.

CLI: `python -m inm.encoder_audit --config CONFIG` with mutually exclusive
`--plan`, `--inventory`, `--task-index N`, `--smoke`, `--summarize-only`.
Default device CPU; explicit CUDA may accelerate replay, not change semantics.
Plan/inventory are dependency-light. Unimplemented actions fail explicitly until
their session lands. Inventory prints JSON to stdout; it does not modify inputs.
Smoke uses tiny synthetic fixtures, CPU, and a separate output path marked
`synthetic: true`. Never use a real input/output directory for smoke.

## Reports and the evidence gate

Each new `tasks/Axx_seed_s/audit.json` contains subject, seed, status,
`synthetic`, `partitions: ["train", "validation"]`, `input_study_ids` with
`baseline` and `legacy`, nonempty objects `alignment`, `clean_checkpoints`,
`probes`, `reconstruction`, plus `checks` with exactly these Boolean fields:
`source_integrity`, `trial_alignment`, `split_isolation`,
`normalization_replay`, `checkpoint_replay`, `probe_controls`, `mask_pairing`.
The fields record executed validations, never defaults. `probe_controls` means
the declared controls ran and software invariants passed, not that a particular
real accuracy threshold was attained. Recovered historical heads remain explicitly
unavailable; that known absence is not a failed replay of a supplied checkpoint.

Save per-sample diagnostic outputs only for train/validation; use stable IDs.
Record original checksums, package versions, new config/source identities,
timings, and numerical tolerances. Write atomically, preserve failures, and reject
changed or corrupt reuse. Never overwrite incompatible audits. Report repeats
then seeds then equally weighted participants; preserve negative effects and
withhold incomplete cohort means. Paired uncertainty, if supplied, resamples
participants and is exploratory. No optimizing from the report.

`report/evidence.json` has schema_name `agfl-encoder-audit-report-v1`, status,
synthetic, partitions, subjects, seeds, input_study_ids, and
`source_files_sha256` (all root `*.py` and every `.py` under `agfl/` and `inm/`,
relative to repo root; SHA-256 of file bytes). `tasks` contains exactly one
`{subject, seed, path, sha256}` record per configured task; paths are relative to
the audit output root. All task metadata must match its report and the files'
digests must verify. The summary also writes human-readable Markdown and CSVs.
Only a complete real 27-task report may pass session 08's runner gate. The gate
checks structure, identities, source bytes, flags, and task hashes; scientific
interpretation remains a review responsibility, not a receipt guarantee.

Sessions 01–07 use synthetic acceptance only. After those sessions, run the
documented real A01/seed-0 audit manually, inspect it, then explicitly run the
remaining tasks. Session 08 reads these outputs and writes a diagnosis. It may
predeclare one bounded future intervention or conclude no intervention is
justified. No real data fitting or architecture change runs inside a coding session.
