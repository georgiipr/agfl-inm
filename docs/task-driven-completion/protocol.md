# Task-driven raw completion protocol

This isolated successor tests whether classification-trained raw Tucker
completion improves electrode-loss robustness of the frozen historical
`spatial_transformer`, whose front end is EEGNet. The fixed declaration is
[`configs/task-driven-completion.json`](../../configs/task-driven-completion.json).
The approved [contract](../../plans/task-driven-completion/CONTRACT.md) governs
implementation. Session 01 supplies declaration, provenance and read-only
planning; it does not implement or run fitting.

Nine participants and seeds 0, 1 and 2 give 27 subject-major tasks. Each task
evaluates five strategies: `zero`, `covariance_frozen`, `covariance_learned`,
`tucker_frozen`, and `tucker_learned`. Only the two learned completers receive
supervised updates: 54 supervised completion fits, zero backbone fits, 27
training-only covariance initializations and 27 training-only Tucker
initializations. The two frozen controls share the initial parameters and
inference operation of their learned counterparts.

## Historical inputs and availability

The sole backbone is the historical Transformer from
`results/encoder-candidates-reproducible-v1`; split metadata comes from
`results/baselines-reproducible-v1`. Historical readers in
`inm/completion_transformer` remain unchanged. Their strict verifier checks
both spatial EEGNet and spatial Transformer, although this successor only
trains/evaluates completion for the Transformer. Both strict verification and
original full-input prediction replay must pass before real completion fitting.
A previous study's successful replay receipt is not a substitute for rechecking
the current inputs and source identity.

Only original T-session training and validation trials participate. No test
arrays, labels or scores may be loaded. Historical 2–30 Hz per-window filtering
and training-only channel mean/population-standard-deviation normalization
remain unchanged. Normalized raw tensors have shape `[B,22,4,250]`; original
electrode-identity masks are Boolean `[B,22,4]`. Completion operates separately
within each window. The inherited historical EEGNet convolutions still span
the full trial after window completion.

Select observed samples with `torch.where` before arithmetic, reject observed
nonfinite samples, malformed masks and empty windows, and make hidden NaN/Inf
irrelevant. Observed samples remain bitwise unchanged. Filled values must reach
the frozen model alongside the original availability flags. Freeze backbone
weights, BatchNorm and dropout in evaluation mode while retaining gradients
with respect to input. The historical replay adapter's `no_grad` path is not
suitable for training completion. Full masks bypass completion exactly;
full-input predictions are invariant and full-input accuracy gain is not an
objective.

## Fixed completion and training settings

Tucker uses `U[22,4]`, `V[250,16]` and per-window `G[4,16]`. Both arms use the
same differentiable observed-entry ridge solve in float64. With `n_observed`
electrodes, the normal-equation ridge is `n_observed * 250 * 0.001`. Initialize
once per task with the existing reconstruction-only `inm.tensor_attention.Tucker2`
fit on training data: 30 epochs, learning rate 0.01, batch size 64, seed equal
to task seed plus 810001, restoring ambient RNG afterward. Copy identical
factors into frozen and learned arms; normalize learned factor columns to unit
length after optimizer steps.

Covariance starts from the training channel second moment `S`, stabilized as
`S0 = S + 1e-6 I`. Both arms use the same SPD parameterization `Sigma = L L^T`,
with lower-triangular free parameters and a positive diagonal given by
`softplus + 1e-6`. Inverse-softplus initialization must reconstruct `S0`.
Missing predictions are
`Sigma_mo (Sigma_oo + 0.001 mean(diag(Sigma)) I)^-1 x_o`.
The learned arm updates its free Cholesky parameters. This is a supervised
nontensor control; positive finite ridge and stable solves are required.

The supervised loss is classification cross-entropy through the frozen
backbone, plus 0.1 times mean squared reconstruction error on artificially
hidden **training** samples, plus `1e-4` times mean squared parameter
displacement from initialization. Displacement uses `U/V` for Tucker and free
Cholesky parameters for covariance. Empty reconstruction targets yield a
differentiable zero. Wholly full batches skip optimizer updates. Use Adam,
learning rate 0.001, batch size 32, maximum 100 epochs, minimum 10 epochs,
patience 20 and gradient norm clip 1.0; no scheduler or optimizer weight decay.

For each training example choose retained count 22/16/6 uniformly. For nonfull
examples choose static/dynamic random loss equally. Use existing
`make_mask_bank`, `partition='train'`, original sample IDs and an epoch/repeat
derived seed, ensuring subsets are disjoint from validation. The learned arms
share batch order and mask schedule. Model and data RNG streams must be scoped
independently; restore Python, NumPy, Torch RNG and thread state, including
exceptions. Initial execution is **CPU, one thread**; CUDA remains unverified.

## Selection exception and reporting

The contract explicitly permits degraded-validation selection for this
successor because full-input metrics are constant. Select the largest mean
validation balanced accuracy over static/dynamic 16/6 electrodes, with five
fixed repeats of each condition. Ties prefer lower mean degraded log loss,
then earlier epoch. Epoch 0 (initialized state) is eligible. Run at least 10
training epochs before patience may stop. Frozen controls have no supervised
selection. Never optimize parameters on validation labels or targets.

Evaluate full 22 electrodes once and the same four degraded conditions with
five repeats each: 21 rows per strategy/task. Selection and final evaluation
use the same degraded validation masks, paired across strategies. Persist
numeric probabilities, labels, sample IDs, masks and hashes, complete initial
and selected completion states, histories, selected epoch and checksums for
independent replay.

The primary contrast is mean degraded validation balanced accuracy of learned
Tucker minus frozen Tucker. Secondary contrasts are learned Tucker minus frozen
covariance, learned Tucker minus learned covariance, learned covariance minus
frozen covariance, and learned Tucker minus zero. Average repeats, then seeds
within participant, then participants equally. The primary screen requires at
least +2 percentage points and positive effects for at least 6/9 participants;
report it separately from the strong-control comparisons. Beating zero alone
does not establish the proposed benefit.

Bootstrap paired participants with 2,000 resamples and seed 20261005; intervals
are exploratory and unadjusted. Preserve negative effects, condition-level log
loss, participant variation and incomplete coverage. Suppress real cohort means
if coverage is incomplete; synthetic evidence cannot enter real results.
Historical and completion checkpoints both reuse validation for selection,
so these results cannot provide independent confirmation, causal recovery or
architecture-superiority claims.

## Dependency-light commands and identity

```bash
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --plan
.venv/bin/python -m inm.task_driven_completion --config configs/task-driven-completion.json --preflight
```

Both commands use only the standard library and create no study output.
`--plan` reports counts; `--preflight` hashes input files using the predecessor's
read-only inventory. Preflight exit 0 means the inventory is complete, **not**
that fitting is ready. Missing files/packages yield exit 2. Even a complete
inventory reports `real_fitting_ready=false`, `historical_verification=not_run`
and `original_full_input_replay=not_run`. Byte identity does not prove semantic
or numerical validity. The strict historical verifier loads NumPy prediction
arrays, so it belongs to the later numerical execution gate, not this
dependency-light preflight. Neither command accepts a fitting or CUDA mode in
session 01.

JSON validation rejects duplicate/unknown/missing keys, nonfinite values,
wrong scalar types, cohort drift and any changed scientific setting. Synthetic
declarations may use subsets of the declared subjects/seeds, while keeping
scientific settings fixed. All paths resolve relative to the config. Default
output is `results/task-driven-completion-v1`. Resolved output paths cannot
overlap source, configuration, data, historical roots or existing other study
results; symlink descendants in an existing output are rejected. Nonempty
output is refused by default. The API's explicit `allow_existing_output=True`
supports read-only inspection only; it grants no resume authorization.

`study_identity` records exact config bytes, resolved config, current Python
source in the root/`agfl`/`inm` (including reused and new packages), installed
scientific package versions, Python version, historical top-level manifests,
per-task metadata, original split/dataset metadata and the original candidate
config. Missing historical files remain explicit nulls. Changed source,
packages or configuration requires a fresh output identity. Later execution
must recheck historical artifact hashes and accept resume only for complete,
verified tasks; foreign, partial, failed or corrupted output must fail closed.

The next sessions implement numerical models, training and execution. All
coding-session verification remains synthetic CPU work. A real pilot,
operational review and cohort require a separate explicit operation.
