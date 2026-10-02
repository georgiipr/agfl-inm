# Accuracy baseline audit — 2026-10-01

## Readiness observed in this checkout

The legacy plan is available without the scientific stack and declares 27
participant/seed tasks with 14 classifier fits each (378 fits). The legacy
configuration is `configs/study.json`; it expects T-session recordings
`A01T.gdf` through `A09T.gdf` under the working-directory-relative `../ml`
path, and writes under working-directory-relative `results/inm-v2`. At audit
time, `/home/kalexu97/Projects/ml` and `results/` are absent. The environment inspector records
package discoverability and file existence without importing numerical
packages or reading recordings. It must be rerun in the intended execution
environment before training.

The supplied `AGFL_PYTHON` is Python 3.13.13. In that interpreter, `torch`,
`numpy`, `scipy`, `scikit-learn`, and `mne` are not discoverable; `tqdm` is
discoverable. The legacy implementation imports PyTorch and NumPy at module
load, uses scipy/scikit-learn/MNE in its supporting pipeline, and requires CUDA
for normal execution. Package availability is environment-specific and this
audit does not install anything. CUDA availability is not probed by the
stdlib-only inspector.

No saved legacy results were present, so there are no observed participant
scores, selected epochs, histories, or calibration records to compare. This is
missing evidence, not zero performance. No SOTA or accuracy-improvement claim
can be made without comparable results under a declared protocol.

## Legacy artifact format

The output root contains `study.json` with `study_id`, source/config identity,
task pairs, and arm declarations. Task artifacts use
`artifacts/A01_seed_0/` naming. `split.json` stores partition indices and split
identity. `calibration.json` stores the `encoder_selection` object, including
`best_epoch`, `selected_validation`, and the validation-only selection rule;
the matching calibration tensor cache is `calibration.pt`. Per-arm records are
`ARMS/<arm>/result.json`, with `status`, task/arm identity, `selection`, and
`metrics` rows. Rows identify `partition` (`validation` or `test`), scenario,
mask repeat/hash, and accuracy, balanced accuracy, macro F1, plus optional
reconstruction NRMSE. Result identities include study, subject, seed, dataset
fingerprint, split ID, and calibration SHA256. Histories are
`encoder_history.json` and `ARMS/<arm>/history.json`; report outputs include
`accuracy_by_run.csv`, `accuracy_by_subject.csv`, `accuracy_table.csv`,
`robustness_table.csv`, `paired_tensor_gain.csv`, `progress.json`, and
`summary.md`.

The stdlib diagnostic handoff schema is:

- `inspect_environment`: `python {version, executable}`, `packages` keyed by
  distribution name with `discoverable`, `config_path`, `config_status`,
  resolved `input_dir` and `output_dir`, `expected_recordings` entries with
  `path` and `exists`, and `readiness`.
- `inspect_legacy`: `output_dir`, overall `status`, `completion`
  (`state`, `tasks_complete`, `tasks_expected`), `study_identity`, `tasks`, and
  `findings`. Each task carries participant/seed status, encoder selection
  labeled `VALIDATION`, split and calibration identity, and per-arm full-input
  scores whose partition is explicitly `VALIDATION` or `TEST`. Missing files
  are unavailable; malformed JSON is surfaced as corrupt.

## Known architectural concerns

The existing concerns remain relevant when interpreting its eventual results:
within-participant T-session splitting does not test cross-session transfer;
the shared encoder is MHA-supervised; calibration uses all channels; missing
placeholders remain attention tokens in the baseline; and core/completion
pooling and model sizes differ. There are only nine participants and many
comparisons. The baseline study is a separate protocol and must not be mixed
with these legacy results.

## Exact evidence still missing

- All nine T-session GDF recordings in the configured input directory.
- A discoverable scientific runtime, including compatible PyTorch, NumPy,
  SciPy, scikit-learn, and MNE; CUDA availability for legacy training.
- Saved complete legacy artifacts with readable calibration and result JSON,
  to establish any empirical full-input baseline/core score or encoder
  validation-selection metric.
- Comparable baseline-first study runs before assessing an accuracy change.

The current audit has not trained a model, read raw GDF contents, or made an
accuracy or SOTA judgment.
