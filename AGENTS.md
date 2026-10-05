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
dataset helpers. The broad `datasets/` ignore pattern at the reviewed revision
hid that tracked directory from ordinary ripgrep listings; the research branch
now anchors recording-directory patterns to the root. Consult Git's tracked
inventory before declaring any package absent.

`python run.py --plan` is dependency-light and should print 27 tasks and 378
classifier fits for the supplied configuration. Actual training requires the
scientific dependencies, CUDA, and `A01T.gdf` through `A09T.gdf`. Do not launch
the full study as a documentation check. No checked-in test suite or Slurm
launcher was found at the reviewed revision; choose checks appropriate to changes.

Build the concept document with `make -C docs/concept`. Review the LaTeX log and
render diagrams when modifying layout. This task added documentation only.
Root `AGENTS.md` is now tracked on the research branch, despite the historical
ignore rule, so the orientation travels with its source snapshot.

## Research branch

Continue the baseline accuracy work on `research/baseline-accuracy`. Read
`docs/baselines/branch-review.md` for the review of George's `origin/main` at
`92b7fc2`. Its end-to-end raw-completion experiment is a separate design;
merging it wholesale would change the protocol and break baseline imports.
The small root-anchored ignore-rule fix was adopted separately. Keep completed
study identities intact and review future contributions individually.

## Approved successor work

The user approved a simpler baseline-first accuracy investigation, planned in
`plans/accuracy/README.md`. Its shared `CONTRACT.md` explicitly permits a separate
baseline package with end-to-end training, early spatial mixing after raw masking,
no-attention models, optional robust validation selection, and T/E session splits.
These exceptions apply to the new baseline study, not the legacy experiment.
Execute one bounded session at a time. Plans, the automation runner, and its
acceptance checker are not session implementation targets.

The user also approved the encoder/class-information investigation in
`plans/encoder-audit/README.md`. Its contract adds a separate analysis package
with immutable historical inputs, train/validation-only checkpoint replay and
fixed probes. Sessions 01–07 implement tools using synthetic checks; real audit
execution follows their runbook, and session 08 requires complete real evidence.
Do not infer authorization to retrain encoders or launch an architecture search.

The user approved planning a fresh five-arm encoder architecture comparison in
`plans/encoder-candidates/README.md`. Execute its bounded implementation sessions
under `CONTRACT.md`: end-to-end local power, spatial filter-bank and compact
Transformer candidates are allowed only in the separate `inm/encoder_candidates/`
package. Preserve legacy code and both earlier plans/runners. Sessions 01–09
use synthetic CPU checks; real training follows the runbook separately, and
session 10 requires complete real evidence. Plans and automation are not session
implementation targets. Historical replay blockers do not block fresh control
training with verified read-only split metadata and a new study identity.

The reproducibility/execution follow-up is planned in
`plans/candidate-followup/README.md`: sessions 01–02 repair RNG control and verify
CLI execution/readiness, session 03 reviews a real pilot, and session 04 reviews
the real cohort. These new plans permit narrowly scoped RNG and execution fixes
inside inm/encoder_candidates only; prior architectures/settings remain fixed.
Real fitting is an explicit separate `scripts/run_candidate_experiments.sh`
operation after verified readiness and pilot review, never a coding-session task.
Preserve earlier workflow receipts and use the new reproducible output identity.

The user subsequently authorized implementing and testing tensor temporal
encoders without an EEGNet front end. This separate successor lives in
`inm/tensor_temporal/`, with the frozen five-arm declaration
`configs/tensor-temporal-screen.json`: three tensor variants, a spectral dense
control, and the existing spatial Transformer reference. The bounded real screen
uses nine participants, seed 0, training/validation only, and its own output
identity. It explicitly permits supervised multilinear spatial mixing after raw
masking and local temporal patching; frozen spectral Tucker factors still use
training data only. `scripts/run_tensor_temporal_experiments.sh` separates pilot,
cohort and reporting. Do not change architectures/settings after observing this
screen or reuse its artifacts under changed source. See
`docs/concept/tensor_temporal.tex` for mathematical definitions and limitations.

The user approved the next matched test of classification-trained spectral
Tucker factors. `inm/supervised_tucker/` permits gradients through an observed-entry
ridge solve and unit-column factor updates only in its supervised arm; the paired
control retains frozen factors. Both arms use the same solver, calibration and
head initialization. `configs/supervised-tucker.json` declares nine participants,
three paired split/training seeds and 54 fits in a new output identity. Full-input
validation selects checkpoints; repeats and seeds are averaged within participant
before cohort aggregation. Preserve all earlier scientific code and artifacts.

The completed task-driven completion study uses `inm/task_driven_completion/`
and its isolated CUDA execution package `inm/task_driven_completion_cuda/`.
The frozen v2 declaration compares zero fill, frozen/learned covariance and
frozen/learned raw Tucker before the same frozen historical EEGNet–Transformer.
It explicitly selects completion checkpoints on mean degraded validation BA,
with epoch zero eligible; full-input predictions are invariant. Do not reinterpret
its validation results as independent test evidence. Shared numeric evidence is
in `evidence/learned-covariance-v2/`; friend-facing review and fresh-run commands
are in `docs/task-driven-completion-cuda/reproduce.md`. The Bash launchers keep
original artifacts immutable, require a new output identity for reproduction,
and independently audit each task. Raw GDF recordings remain external.
