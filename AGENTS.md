# Project orientation for coding assistants

## Start here

Read `docs/README.md`, `docs/onboarding.md`, and `docs/investigation.md` before
changing research behavior. The mathematical introduction is
`docs/concept/agfl_concept.pdf`; its editable source is `docs/concept/agfl_concept.tex`.
The investigation describes source revision `84ae174`, reviewed on 2026-10-01;
recheck facts when code or configuration changes.

## Purpose and boundaries

AGFL-inm compares EEG classification under known electrode loss using MHA and
Performer, with baseline frozen EEGNet-derived features and a training-fitted
Tucker-2 representation. `inm/` owns the experiment; `agfl/` is the model and
attention library. `run.py` is the entry point; `configs/study.json` is the
declared experiment. Signal Transformer exists in the library but is outside
this experiment. Do not infer an expansion of the acronym AGFL.

The original proposal in `references/` and the implementation are different
designs. The supplied experiment trains participants individually, splits their
T sessions approximately 60/20/20, uses frozen learned features, and applies
spatial attention per window without temporal position embeddings. It does not
implement the proposal's subject-disjoint folds or Nyström attention.

## Research invariants

- Fit normalization and Tucker factors using training trials only. Select
  supervised epochs on full-input validation; never select using test results.
- Keep original electrode identities and Boolean masks `[B,22,P]`. Select
  observed values with `torch.where` before arithmetic; hidden NaNs must not leak.
- Keep filtering and convolution local to each channel/window. Freeze BatchNorm
  and dropout behavior in the shared encoder before extracting cached features.
- Keep Tucker factors frozen as buffers during classifier training. Core solves
  use observed entries only and the normalization implies ridge `n_observed*F*lambda`.
- Pair masks, splits, trial order, and calibration across comparison arms.
  Mixed training uses 22/16/6 electrodes and random losses; 11 and spatial losses
  are evaluation-only. No all-missing window is allowed.
- Disclose baseline missing placeholders: they participate in attention but are
  excluded from baseline spatial pooling. Completion/core pooling differs.
- Average mask repeats, then seeds within participant, then participants equally.
  Preserve negative effects and incomplete results; do not invent measurements.
- Preserve result provenance. Changed source/packages/configuration require a
  new output directory rather than reuse of incompatible study artifacts.

## Practical checks

Use `git ls-files` or `rg --files --no-ignore agfl/datasets` when reviewing the
dataset helpers: the existing broad `datasets/` ignore pattern hides that tracked
directory from ordinary ripgrep listings. Consult Git's tracked inventory before
declaring any package absent.

`python run.py --plan` is dependency-light and should print 27 tasks and 378
classifier fits for the supplied configuration. Actual training requires the
scientific dependencies, CUDA, and `A01T.gdf` through `A09T.gdf`. Do not launch
the full study as a documentation check. No checked-in test suite or Slurm
launcher was found at the reviewed revision; choose checks appropriate to changes.

Build the concept document with `make -C docs/concept`. Review the LaTeX log and
render diagrams when modifying layout. This task added documentation only.
Root `AGENTS.md` is locally ignored by the pre-existing `.gitignore`; explicitly
include it when sharing the orientation package if desired.

## Approved successor work

The user approved a simpler baseline-first accuracy investigation, planned in
`plans/accuracy/README.md`. Its shared `CONTRACT.md` explicitly permits a separate
baseline package with end-to-end training, early spatial mixing after raw masking,
no-attention models, optional robust validation selection, and T/E session splits.
These exceptions apply to the new baseline study, not the legacy experiment.
Execute one bounded session at a time. Plans, the automation runner, and its
acceptance checker are not session implementation targets.
