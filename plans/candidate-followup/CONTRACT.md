# Reproducible candidate execution contract

Continue the five-arm encoder candidate study. Preserve every architecture,
training budget, loss, partition, probe and comparison from
[the candidate contract](../encoder-candidates/CONTRACT.md). The previous nine
coding sessions are complete, but no real candidate fits were found at review.
All 63 tests passed; a separate synthetic check exposed uncontrolled CPU dropout
RNG despite fixed model seed and initial weights. Do not invent real scores.

This follow-up fixes reproducibility, verifies execution, and reviews actual
pilot/cohort evidence. It does not add candidates or tune hyperparameters.
Default coding sessions are 01–02. Session 03 requires a real five-arm A01/seed0
pilot; session 04 requires 27 real tasks /135 neural fits /108 CPU probes.
The explicit experiment launcher, not a coding agent, runs those fits between
sessions. No model calls or real fitting are performed by preparing these plans.

## Preserve previous work

Do not edit earlier plans, runners, receipts, scientific outputs or configs.
Use `configs/encoder-candidates-reproducible.json`, with a new name and
output_dir `../results/encoder-candidates-reproducible-v1`. Copy every scientific
setting from configs/encoder-candidates.json unchanged. Name may become
encoder_candidates_reproducible_v1. Same split source and recordings. The new
output is required because reproducibility fixes change source identity.

Bounded changes to inm/encoder_candidates and its tests are allowed in sessions
01–02. No old baseline/legacy code changes. Source/config/packages freeze when
session02 writes readiness.json; sessions03–04 change documentation/tests only.
All input provenance, raw masking, hidden-NaN invariance, train-only fitting,
validation-only selection and equal participant aggregation remain mandatory.

## Reproducibility policy

A declared seed must determine initialization, batch order and training-time
randomness independently of ambient Python/NumPy/PyTorch RNG, arm execution order,
and earlier fits in the same process. Isolate fit RNG and restore caller state,
even on exceptions. Keep data-order RNG independent of parameter count and dropout.
Preserve identical shared-head/front-end initialization across matched arms.

Use CPU torch RNG seeding and selected-device CUDA RNG seeding in a scoped
context, with a documented deterministic seed derivation from subject/task seed
and a constant stream offset. Do not derive seeds from Python hash(), elapsed
time, filesystem paths or arm position. Avoid touching unrelated CUDA devices
when only CPU is requested. Check existing constructor fork_rng/manual_seed use:
CPU initialization must not accidentally reseed CUDA generators globally.

Deterministic kernels/settings must be scoped/restored and recorded. On CUDA,
set CUBLAS_WORKSPACE_CONFIG=:4096:8 before any CUDA context, disable benchmark
selection, and use deterministic operations or fail clearly; never silently
relax settings to make a run finish. No promise of bitwise equivalence across
CPU/GPU, devices, package versions or hardware. Establish repeatability on the
same device/environment. Record RNG policy/version, effective seeds, device,
versions and deterministic settings in checkpoint/result provenance so reuse
cannot mix policies. Keep full checkpoints and numeric probe artifacts.
Each fit result also records execution_device as cpu or cuda, reflecting the
actual training device. Pilot and cohort must use the same verified device type.

Required synthetic regression: same task/arm seed and initial state, different
ambient RNG histories, two fresh directories and two actual short fits must
produce identical selected CPU states, histories excluding timings/paths, and
probabilities. Also compare alone versus after a different-arm fit, shuffled
execution order, fresh subprocess execution, different seeds, caller RNG-state
restoration and exception restoration. Repeat for all five models using tiny
budgets; do not compare timing fields or raw torch.save container bytes.

CPU tests are mandatory without skips. CUDA is a separately reported capability:
run tiny numerical repeatability checks only if accessible; otherwise list it
as unverified, not a passed test. The real launcher permits only devices in the
readiness artifact's verified_devices. GPU testing here is synthetic verification,
not permission to train real EEG in a coding session.

## Execution and readiness

Review public config/CLI resume. Currently load_config rejects nonempty outputs
and __main__ uses a temporary-config workaround. Provide a read-only validation
path for a valid existing output; task/fit writers still reject incompatible,
partial or corrupt artifacts. Never bypass provenance or rewrite manifests.
Avoid replacing outputs or using a new source/config identity for each task.
Test two sequential task indices and a completed-task resume in separate CLI
processes with the SAME config and output. Test summary after populated output,
changed source/config/package rejection, failed fits, checksum tampering and
synthetic exclusion. Keep --plan/--preflight numerical imports lazy.

Session02 performs all candidate tests and tiny all-five-model smoke, checks the
real A01 split/recording loading without fitting, and saves logs under
`docs/candidate-followup/validation/`. Do not let logs include EEG arrays or
held-out metrics. Real data loading for verification is allowed; real training
and architecture decisions are not part of session02.

Write `docs/candidate-followup/readiness.json` with status ready only after all
mandatory checks pass. Required fields:

- status: ready; verified_devices: [cpu] and optionally cuda after actual checks.
- study_id, config_sha256, source_files_sha256, packages copied from
  protocol.study_identity on the final new config/source/environment.
- checks: exactly rng_independence, repeat_fit_equivalence, all_five_smoke,
  pilot_input_verified, all true from executed checks.
- artifact_sha256: nonempty mapping of paths relative to docs/candidate-followup
  to saved verification log hashes. Include command outcomes and interpreter.

Freeze source before calculating readiness; rebuild it after any source/config/
package correction. Documentation-only updates do not change the scientific
source map. The launcher checks readiness hashes and current identity before
any real task, and each subsequent cohort task. Receipts alone do not authorize
an incompatible fit. Software completion with missing real inputs is allowed,
but status ready must not be fabricated: leave readiness status blocked and
explain missing inputs in handoff. Numeric test failures block coding completion.

## Pilot and cohort

Run only the explicit command `bash scripts/run_candidate_experiments.sh --pilot`
after sessions01–02. Default device cpu; explicit --device cuda requires verified
CUDA readiness. There is no default training action, no automatic retry and no
model call in the experiment launcher. Its workspace lock is shared with all
coding runners. Logs are separate from immutable scientific inputs. A pilot
contains five neural fits and four local probes; inspect it before continuing.

Session03 verifies A01/seed0 task, checkpoint/history/prediction/probe hashes,
source/config/packages, selected epoch consistency, sample/mask pairing,
normalization and RNG metadata. Replay clean predictions where appropriate;
no training, no changes to source or tolerances. Judge operational validity,
not whether one participant happens to favor a candidate. Write
`docs/candidate-followup/pilot-review.md` and pilot-review.json:

- decision: proceed or stop; summary: nonempty explanation.
- blockers: [] for proceed, concrete unresolved operational issues for stop.
- study_id: current study; pilot_sha256: exact task.json bytes.
- device: cpu or cuda, matching execution_device in every pilot fit.

A completed review may correctly conclude stop. That is not a software failure,
but it blocks cohort execution. Never use a positive score to bypass bad artifacts.
The new helper checks the actual pilot task directly; it does not fabricate a
complete cohort evidence report or depend on partial report task listings.

Only after a verified proceed review, explicitly invoke --cohort. It runs
indices1–26 sequentially and summarizes once. Existing complete compatible tasks
may be verified and skipped by the experiment code; partial/failed ones stop.
Do not delete or overwrite failures. Repairs require a reviewed fresh scientific
output identity and updated readiness/gate paths, not just a fresh log directory.
This fixed launcher deliberately targets the reviewed reproducible-v1 config.

Session04 reads full evidence at
results/encoder-candidates-reproducible-v1/report/evidence.json and the underlying
27 tasks. It applies the original +2pp BA and 6/9 positive-participant screening
rule, with all three contrasts, paired participant-bootstrap intervals, train/
validation gaps, ordered/shuffled probes and secondary raw-loss diagnostics.
No test-based selection or new fitting. A negative/inconclusive result is valid.
It may propose at most one independently confirmed next experiment. It does not
claim strong classification or Tucker benefit without the corresponding evidence.
