# Fixed scope: published-checkpoint covariance study

This successor uses a published classifier and its audited native preprocessing.
These exceptions apply only to the new package, never to historical studies.
The user's current request is planning; real execution is authorized by the
opening message in README when sent in the execution session.

## Scientific interpretation

Evaluation role: `exploratory_published_checkpoint_robustness`.
Prior project E-outcome exposure remains `unknown`; independently record whether
the external checkpoint used E for training, preprocessing, epoch selection or
run selection. Known test-guided selection and unknown provenance remain visible
in every report. Neither is repaired by hiding electrodes at evaluation time.
Do not manufacture a no-access attestation or a confirmatory significance claim.

The primary question is learned-versus-fixed covariance on static electrode loss.
A positive result supports portability to the tested classifier. It does not
establish SOTA robustness against untested methods. A null result is retained.

## Checkpoint admission, before outcome inspection

First candidate: official EEG-ATCNet nine-subject `results/saved models/run-1`
weights. Pin an exact repository revision, URLs, license and file SHA256 values.
Files named `.h5` are evidence of availability, not proof of model identity,
preprocessing compatibility, evaluation validity or reproduced performance.

Audit architecture, framework/custom layers, expected inputs, class mapping,
participant/session identities, checkpoint training/selection history, and
normalization state. Resolve any missing native configuration from code/history
and checkpoint tensors. Never guess ambiguous settings from the best accuracy.

Use the single complete run-1 set if compatible. Do not mix best-per-subject runs.
If it fails structural admission, inspect official TCFormer assets once as a
fallback, selecting a complete compatible BCI IV-2a set by documented run/seed
order, not observed performance. Do not open an unlimited checkpoint search.
If neither supplies an auditable compatible set, stop with a useful audit and
reproduction recommendation. New backbone training is outside this authorization.

Interpretation grades:

- A: evidence supports T-only training/calibration/selection. Still exploratory
  because current project outcome exposure is unknown.
- B: E-guided selection or unresolved selection history, but complete compatible
  checkpoint/data identity. Permitted as descriptive robustness evaluation;
  prominently disclose selection bias and no independent generalization claim.
- C: evaluation trials were used for supervised weight fitting, or checkpoint,
  subject, labels or preprocessing cannot be resolved. Exclude from this study.

Freeze admission before local accuracy inspection. Low reproduced accuracy is
a result or a discrepancy to investigate, not permission to pick another model.

## Protected scope and provenance

New implementation only in `published_covariance/`,
`tests/published_covariance/`, `configs/published-covariance.json`,
`scripts/run_published_covariance.sh`, and `docs/published-covariance/`.
Supervisor owns `plans/published-checkpoint-covariance/` and
`.session-runs/published-checkpoint-covariance/`.
Keep downloaded source/weights/data under
`.session-runs/published-checkpoint-covariance/assets/` with origin manifests.
New real output: `results/published-covariance-v1/`.

Preserve all pre-existing source/config/results/receipts/plans and untracked work,
including the concurrent exploratory E successor. No existing package imports
may be monkey-patched. Reuse pure helpers read-only or adapt the minimum necessary
code under the new package, retaining attribution and differential tests.
Record the original source of adapted covariance equations and implementation.

Prefer the checkpoint's native framework to an unverified conversion. If a
different runtime is required, create `.venv-published-covariance/` and freeze its
versions without changing the existing `.venv`. Install only needed dependencies.
A port requires both forward and input-gradient parity with native inference.
Freeze the backend decision before real covariance fitting.

Study identity binds new/reused source and plans, external revision/assets,
configuration, actual packages/device, raw data and labels, native preprocessing
and normalization, trial IDs/exclusions/splits, masks, checkpoint tensors and
selection disclosure. Source/config/runtime changes require a fresh output
identity, never reuse of incompatible artifacts. No commits/pushes or external
messages are part of this workflow.

## Native preprocessing and electrode-loss semantics

Preserve the checkpoint's actual sampling rate, input duration/shape, electrode
order, units, filtering, reference, normalization and artifact policy. Do not
impose the old 22×4×250 input, 2–30 Hz filter or flags on an incompatible model.
The exact native recipe must be sealed in the audit before real fitting.

Official ATCNet preprocessing uses a MATLAB recording structure that differs
from the official competition's label-only `AxxE.mat` files. Never mistake one
for the other. Obtain the author's documented recording format from its trusted
source, or verify a GDF adapter against native recording arrays/cue alignment.
Record exact sample offsets and indexing conventions, including any one-sample
discrepancies; do not silently repair a native recipe while calling it identical.

Use the checkpoint's saved normalization, or reconstruct its documented T-only
normalization exactly. It may have used all T trials, including our later
covariance-validation subset: disclose this historical exposure. No new transform
is fitted on E. If a checkpoint requires E-fitted normalization, it cannot be
silently replaced; stop admission and explain the protocol conflict.

Primary loss model: a subset of electrodes is unavailable for the entire trial
and all temporal support needed by preprocessing. Mask shape `[B,22]`, broadcast
over native time samples; stable original channel identity. Evaluate full22 once,
static16 and static6 with five repeats each. Dynamic losses are deliberately
deferred because temporal filtering can leak hidden intervals into observations.
This does not test electrodes permanently absent from the calibration session.

Prove that retained preprocessed channels do not depend on hidden raw channels.
Channel-local filtering plus saved per-channel normalization allows completion
in native normalized input space. A spatial reference/alignment that mixes hidden
electrodes before masking invalidates that equivalence; do not use preprocessed
complete EEG to leak hidden samples. Such a pipeline requires a separately
reviewed mask-aware design rather than an unannounced preprocessing change.

No mask flags are appended to a published classifier that does not accept them.
The completer alone consumes masks. Classifier input shape and weights stay exact.

## Participants, splits and masks

A01–A09, one published classifier per subject, covariance seeds 0/1/2: 27 tasks.
Three seeds vary covariance T split, training order and masks; they are NOT
independently pretrained classifier seeds. Report this distinction prominently.

For each native eligible T trial set, construct a deterministic stratified 80/20
covariance train/validation split: original chronological IDs, classes 0..3,
NumPy `Generator(PCG64(seed))`, permute each class in ascending class order,
first `max(1, floor(0.2*n_class))` to validation, rest to training; return each
partition in original trial order. Fail if any training class becomes empty.
Persist IDs, indices and split hashes. Do not shuffle or select E trials.

Fit covariance initialization and supervised covariance only on the new training
partition. Validation selects its epoch. The pretrained backbone may already
have seen all T labels; covariance validation is not held out from that backbone.
Do not call these T scores generalization evidence.

Masks: choose observed subsets uniformly without replacement, keyed deterministically
by study namespace, subject, seed, original trial ID, partition, epoch/repeat and
retained count. Use disjoint subset pools for train/validation/E, equal pools
across arms, with no all-missing input. Define and test the exact key encoding
and generator in the configuration before real fitting; never use Python hash().
Read-only reuse of the existing mask generator is preferred if its static masks
can be broadcast over arbitrary native input length without changing semantics.

## Completion and training declaration

Arms: `zero`, `covariance_fixed`, `covariance_learned`; same frozen classifier.
For native normalized training samples x(t) in R^22, S is their uncentered
second moment, averaged across covariance-training trials and time samples.
Initialize Sigma = S + 1e-6 I, with a 253-parameter float64 lower-triangular L,
softplus diagonal plus 1e-6, Sigma = L L^T. Fixed and learned start identically.

For observed O and missing H:

    rho = 0.001 * mean(diag(Sigma))
    completed_H = Sigma_HO solve(Sigma_OO + rho I, x_O)

Use a differentiable solve, never an explicit inverse. Preserve observations
bitwise with safe selection before arithmetic; hidden NaN/Inf cannot enter the
solve or preprocessing. Reject observed nonfinite inputs and empty masks.
Full input returns the original native tensor unchanged. Generalize time length
in the new implementation; the old completer hard-codes four 250-sample windows
and cannot be applied blindly to the published input.

Train covariance from initialization for each task, never reuse historical
Transformer-trained covariance. Frozen backbone includes parameters, BatchNorm
statistics and dropout/evaluation behavior. Gradients through its input remain
enabled. Only the covariance optimizer may update state.

Loss = CE(f(completed_x), y) + 0.1 * hidden_sample_MSE(completed_x, x)
       + 1e-4 * mean((free - initial_free)^2).

MSE averages artificially hidden normalized entries only; an empty set gives
differentiable zero. Full-input CE is unnecessary because the frozen classifier
makes it constant with respect to covariance. Keep probabilities/logits and
cross-entropy conventions consistent with the native model; do not apply a
second softmax to probabilities.

Adam lr=0.001, betas=(0.9,0.999), epsilon=1e-8, no weight decay or scheduler,
batch32, clip covariance gradient global norm at1, max100 epochs, min stopping
epoch10, patience20. Training retained count is uniform22/16/6 per example,
whole-trial static masks. Epoch0 is eligible. Select highest mean static16/6
validation BA across five fixed repeats, tie lower corresponding log loss, then
earlier epoch. No E selection. Report epoch0 selections and budget saturation.

## Evaluation, metrics and claims

Seal all 27 selected covariance checkpoints before this study's E evaluation.
Use official E labels and original cue IDs before exclusions; preserve native
trial inclusion policy and verify label/source alignment. Evaluate all arms on
the same retained trials and mask bank. Full-input reference runs once per subject
and remains identical across covariance seeds and arms.

Primary: learned-minus-fixed E BA averaged equally over static16 and static6.
Secondary: learned-minus-zero; per-condition BA, ordinary accuracy and log loss;
absolute degradation from full-input BA; parameter count and measured completion
latency separated from classifier inference. Do not report old four-condition
degraded BA as directly comparable to this two-condition endpoint.

Aggregate repeats, then seeds within participant, then participants equally.
Participant-paired bootstrap95% intervals: 2,000 draws, PCG64 seed20261006.
Intervals are exploratory and unadjusted; no confirmatory p-value threshold or
binary SOTA-success rule. Report all participant effects and negative results.
Suppress cohort aggregates when any required task/arm/cell is missing.

Save ordered IDs/labels, normalization, split/mask banks, checkpoints and full
training histories, probabilities, per-cell metrics, source references and
exclusive manifests. Independent audit recomputes probabilities/metrics, frozen
states, full bypass and masking invariants; report aggregation has a separate
independent numeric check. Native/wrapped parity is not inferred from similar BA.

## Execution boundaries

Real fitting and scoring are supervisor operations, not coding-worker tasks.
Launcher modes: plan/preflight/smoke/reference/pilot/review-pilot/train-cohort/
evaluate-cohort/audit/summarize. No default execution. Plan is dependency-light.
Pilot fits A01 seeds0/1/2 on T; no E scoring. Train-cohort reuses audited pilot
tasks and seals all selected states. Evaluate-cohort refuses incomplete T fitting.

Hold the shared `.session-runs/accuracy.lock`; bound child process groups with
GNU timeout, TERM then KILL after20s. Pilot fit ceiling3600s per seed; audits and
reference ceiling600s. Benchmark real pilot time/memory and record any prospective
cohort timeout adjustment for operational reasons before launching the cohort.
Never change epochs/settings in response to accuracy or silently retry a timeout.

Keep logs/exits and identity-bound pilot/fit-completion receipts. Audit compatible
completed tasks on resume. Partial/corrupt/foreign/incompatible outputs stop;
never delete or overwrite them. Repeat summaries verify equality. CPU fallback
is never silent; real training requires verified CUDA. End with actual process
status, not a claim of a background watchdog after supervisor exit.
