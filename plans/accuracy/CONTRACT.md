# Shared implementation contract

## Scientific scope and compatibility

Build `inm/baselines/` as a small study package alongside the original experiment.
Do not turn the existing model registry into a plugin framework. Reuse loading,
split persistence, masks, metrics, and reproducibility helpers where compatible.
Keep `run.py --plan` and `configs/study.json` unchanged at 27 tasks / 378 fits.
Adding Python files changes legacy source hashes: previously recorded studies
must retain their original source snapshot; new executions use new output dirs.

The new package targets diagnosis and strong baselines, not promised SOTA. Default
comparisons use identical splits, preprocessing, budgets, and masks. The reference
is an EEGNet architecture reproduction, not a claim to reproduce any paper's
score unless its complete protocol is matched. Within-session and cross-session
results must never be merged. No automatic cross-subject transfer or foundation
model work is included.

Legacy invariant exceptions apply ONLY in the new baseline package:

- End-to-end training is allowed; the new CNN need not freeze its encoder.
- Learned temporal/spatial mixing may occur after unavailable raw samples have
  been selected out. Full-input feature caches cannot be masked after mixing.
- Epoch selection may use a predefined robustness validation score.
- The new package permits no-attention classifiers and session T/E splits.

For the old tensor experiment and frozen-feature follow-up, locality, frozen
factors, train-only fitting, and matched masks still apply exactly as documented.

## Proposed file and API boundaries

Keep imports lazy so planning/preflight work without NumPy or PyTorch.

| Module | Stable interface |
|---|---|
| `diagnostics.py` | `inspect_environment(config_path=None) -> dict`; `inspect_legacy(output_dir) -> dict` |
| `protocol.py` | `load_config(path) -> dict`; `tasks(cfg) -> list[tuple[int,int]]`; `arms(cfg) -> list[dict]` |
| `data.py` | `prepare_subject(cfg, subject, seed, directory) -> PreparedData` |
| `eegnet.py` | `EEGNetClassifier(...).forward(raw, mask=None) -> logits[B,4]` |
| `training.py` | `fit_classifier(model, prepared, cfg, arm, directory, device) -> dict` |
| `study.py` | `run_task(cfg, task_index, device) -> dict`; `run_smoke(cfg, output_dir) -> dict` |
| `covariance.py` | `CovarianceClassifier.fit(x,y)`; `.predict_proba(x)`; serializable fitted state |
| `reporting.py` | `summarize(cfg) -> dict`, writes `report/summary.md` and CSVs |
| `__main__.py` | CLI dispatch only, without module-level numerical imports |

`PreparedData` carries raw preprocessed signals `[N,22,1000]`, integer labels,
stable IDs, split indices, channel names, metadata, training-only normalization
statistics, and a data fingerprint. Keep the unnormalized filtered signals
available for the classical pipeline. Do not normalize using validation or test
statistics, including within sklearn preprocessing.

These signatures are contracts rather than finished code. Small additional
keyword arguments are permitted when required; record them in the handoff and
update the new package's implementation documentation, not these plan files.

## Baseline configuration and CLI

Create `configs/baselines.json` with schema name `agfl-baselines-v1` (independent
of legacy schema 2), subjects 1–9, seeds 0–2, data in the repository's sibling
`ml/`, and results in the repository's `results/baselines-v1/`. Relative paths
resolve from the config's directory: use `data_dir: ../../ml` and
`output_dir: ../results/baselines-v1` in the JSON. Document the resolved paths
explicitly. Do not silently reuse legacy CWD-dependent semantics.

Support `within_session` (T only, stratified 60/20/20) and `cross_session`
(train/validation from T at 80/20, E exclusively test). Ship within-session as
the default for a direct diagnostic against current code, plus
`configs/baselines-cross-session.json` for the later benchmark. Both use the same
window-local 2–30 Hz filtering initially. External E labels must be supplied
explicitly via the existing loader contract. Never infer or guess those labels.

Default matrix: `eegnet_reference/full`, `masked_eegnet/full`,
`masked_eegnet/mixed`, `covariance/full`: 4 arms × 27 tasks = 108 fits.
Reference and masked EEGNet share convolutional architecture; the reference
omits mask conditioning. All neural arms can be scored under raw masking; the
classical arm is full-input only. Report that coverage difference explicitly.
Until session 07, reject attempts to run the classical arm with an actionable
message; never silently substitute another model.

CLI: `python -m inm.baselines --config CONFIG` with one of `--plan`, `--preflight`,
`--task-index N`, `--smoke`, `--summarize-only`. Support `--device cpu|cuda` for
execution; default CUDA for real tasks, force CPU in smoke. Smoke uses synthetic
data, a separate output path, at most two epochs, and explicit `synthetic: true`;
it must never reuse or emit records into a real study directory.

## Model and training defaults

EEGNet reference: temporal convolution kernel 64, F1=8, depth multiplier D=2,
F2=16, depthwise spatial kernel `[22,1]` BEFORE ELU/pooling, temporal pooling 4
then 8, separable temporal kernel 16, dropout 0.5, flatten and linear logits.
Constrain spatial weights and final classifier consistently with the documented
EEGNet reference. Keep tensor shapes explicit and use a computed shape, not a
dummy training-mode forward that updates BatchNorm.

For both neural variants, validate Boolean `[B,22,4]` masks, reject empty windows,
and use `torch.where` on raw `[B,22,4,250]` before ALL learned mixing. Arbitrary
hidden values/NaNs must not affect outputs or gradients. The masked variant adds
the flattened 88 observation flags to its classifier input. Both variants use
identical observed preprocessing and mask banks. Evaluation must disable
dropout and BatchNorm updates. Early spatial filtering means cached full-channel
activations must not be reused as masked inputs.

Initially reuse the legacy CE/AdamW budget: up to 250 epochs, batch 32, LR 0.001,
weight decay 0.0001, 10-epoch warmup/cosine, clip norm 1, minimum 75 epochs,
patience 50. Keep these configurable for smoke fixtures. Compare clean train and
validation metrics under `eval()` in diagnostics; training-time dropout and mask
losses are not directly comparable with clean evaluation metrics.

Selection policies: `full` uses full-input validation BA then full validation log
loss; `robust` averages validation BA equally over full, random-static 16,
dynamic-random 16, random-static 6, dynamic-random 6, with fixed validation banks
and mean validation log loss as tie-breaker. Default `full` for the initial
controlled comparison; robust selection is an explicit separately identified
ablation. NEVER include test, 11-channel, or spatial-loss scores in selection.
Training mixed masks use the existing 22/16/6 random-mask protocol.

Save selected weights, exact model constructor settings, train normalization,
selection policy/score, histories, class/channel order, and source/config/data/
split identity. Reloaded predictions must match the selected model. Use a
weights-only compatible tensor/primitive checkpoint format where practical.

## Records and reporting

Per-task paths: `artifacts/A01_seed_0/{dataset.json,split.json}` and
`ARMS/<arm_name>/{history.json,checkpoint.pt,result.json}` (classical fitted state
can use its own clearly documented format). Study manifest must contain all
new package Python source, settings, package versions, and input checksums.
Use atomic writes and reject incompatible or incomplete reuse. Preserve failures.

Each result records subject/seed/arm/protocol, selected epoch, parameter count
where meaningful, timings, full/degraded metric rows, exact evaluation mask
hashes, sample counts, and dataset/split/source identities. Add an explicit
`synthetic` flag. Report aggregation averages repeats → seeds → participants.
Do not mix protocols, budgets, preprocessing, selection policies, or synthetic
and real results. Full-input comparisons with the classical arm are permitted;
degraded comparisons require equal mask coverage and matching hashes. Use BA,
accuracy, macro-F1, and per-class recalls. Suppress incomplete cohort means.

## Acceptance and data-dependent limits

Tests live in `tests/baselines/test_<topic>.py`, using `unittest`. No skips for
missing dependencies: fail or report the session blocked. A source-only parse
does not count as numerical validation. Each session's manifest declares the
required artifact paths and test patterns; the runner checks these externally.

Sessions 11–12 require a **real, complete A01/seed-0 pilot** under the declared
baseline config as an operational prerequisite, plus readable histories and
reports. This does not establish a scientific effect. Configuration selection
must use validation-only evidence across the declared cohort, not one favorable
participant or test scores. These sessions implement ablation support; they do
not run a large search. All-nine-participant, multi-seed evaluation comes later.
