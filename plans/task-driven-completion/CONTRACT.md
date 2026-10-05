# Task-driven completion implementation contract

Approved successor, 2026-10-05. Implement a new, isolated study in
inm/task_driven_completion/, tests/task_driven_completion/,
configs/task-driven-completion.json and docs/task-driven-completion/.
Preserve ALL existing scientific code, configurations, plans, runners, receipts
and results including untracked files. Supervisor owns this plan directory and
.session-runs/task-driven-completion/. No worker edits orchestration/acceptance.
Branch: research/baseline-accuracy. Read root AGENTS.md and project orientation.

## Scientific question and fixed design
Can classification-driven raw Tucker completion improve electrode-loss robustness
of a frozen historical spatial_transformer (EEGNet front end)? Five strategies:
zero, covariance_frozen, covariance_learned, tucker_frozen, tucker_learned.
One backbone only, from results/encoder-candidates-reproducible-v1; original
splits from results/baselines-reproducible-v1. Reuse strict historical verification
and replay readers READ-ONLY from inm/completion_transformer. These verify both
historical backbones; new study evaluates/trains completion only for Transformer.
Nine subjects, seeds 0/1/2, 27 tasks, 54 supervised completion fits; no backbone
fits. Train/validation only, no test arrays, labels or scores. Full-input original
prediction replay is required before real fitting; fail closed if unavailable.
Separate output results/task-driven-completion-v1. Paths relative to config.
CPU one thread initially; CUDA is unverified and must not be offered as ready.

Raw normalized inputs [B,22,4,250], original Boolean masks [B,22,4]. Historical
2-30 Hz per-window filtering and training-only channel normalization unchanged.
Complete independently inside each window. Use torch.where to select observed
values BEFORE arithmetic. Hidden NaN/Inf invariance; reject observed nonfinite,
malformed masks and empty windows. Observed samples stay bitwise unchanged.
Original availability flags must reach classifier even with filled input.
Whole-trial historical EEGNet convolutions remain as inherited successor behavior.
Backbone weights, BatchNorm and dropout stay frozen/eval, but input gradients
must flow. Do not reuse no_grad-decorated replay adapter for training. Full masks
bypass completion exactly, so full-input predictions are invariant and provide
no completion gradient. Never present full-input accuracy gain as an objective.

Tucker: U[22,4], V[250,16], per-window G[4,16]. Same differentiable float64
observed ridge solve in BOTH arms; normalized residual implies penalty
n_observed*250*0.001. Initialize once per task with existing train-only Tucker2
reconstruction fit: epochs30, lr.01, batch64, seed=task seed+810001; restore RNG.
Copy identical initial U/V into frozen/learned models. Supervised factors have
unit columns after optimizer steps. No new ranks, temporal model or head.

Covariance: training channel second moment S, stabilized initialization
S0=S+1e-6*I. Both arms use identical Cholesky-based SPD parameterization
Sigma=L L^T with positive diagonal (softplus plus 1e-6); choose inverse softplus
initialization to reconstruct S0. Conditional mean missing prediction uses
Sigma_mo (Sigma_oo + .001*mean(diag(Sigma))*I)^-1 x_o.
Frozen/learned arms share exact initial state and inference; learned lower-triangle
parameters can update. Positive finite ridge and stable solves mandatory.
This is a supervised nontensor control, not a tensor method.

Supervised completion loss: classification CE through frozen backbone +
0.1*mean squared reconstruction error on artificially hidden TRAINING samples
+1e-4*mean squared parameter displacement from initial completion parameters.
For Tucker the displacement is on U/V; covariance on its free Cholesky parameters.
Empty reconstruction targets yield differentiable zero, never NaN. Skip optimizer
updates for wholly full batches (no classification/completion signal).
Adam, lr.001, batch32, maximum100 epochs, minimum10, patience20, gradient clip1.
No scheduler, no weight decay beyond explicit anchoring. Same batch order and
mask schedule across learned arms; independent scoped model/data RNG streams,
restore Python/NumPy/Torch RNG/thread state including exceptions.
Training masks per example: uniformly choose 22/16/6 retained; for nonfull choose
static/dynamic equally. Use existing make_mask_bank partition='train', sample IDs,
epoch/repeat-derived seed, so subsets are disjoint from validation; fixed across
paired arms. Never optimize on validation targets or labels.

EXPLICIT successor selection exception: full-input metrics are constant, so
select completion epochs by mean degraded validation BA over static/dynamic 16/6,
five fixed repeats each. Tie: lower mean degraded log loss, then earlier epoch.
Include epoch0 initialized state as eligible; require minimum10 training epochs
before patience may stop. Frozen controls have no supervised selection.
Evaluation full22 once and the same four degraded conditions, five repeats each.
Training masks and selection/evaluation masks paired across strategies. Persist
numeric probabilities, labels, sample IDs, masks/hashes, full initialization and
selected states, histories, selected epoch and checksums for independent replay.

Primary: tucker_learned minus tucker_frozen in mean four degraded validation BA.
Secondary: learned Tucker minus frozen covariance; learned Tucker minus learned
covariance; learned covariance minus frozen covariance; learned Tucker minus zero.
Average repeats, then seeds within participant, then participants equally.
Screen >=2 pp and >=6/9 positive participants for primary; report separately from
strong-control comparisons, never promote merely by beating zero. Bootstrap paired
participants 2000 resamples, seed20261005; exploratory, unadjusted intervals.
Preserve negative effects, per-condition log loss, variation, incomplete coverage.
Suppress real cohort means when coverage incomplete; synthetic cannot enter real
reports. Reused validation selected historical and completion checkpoints, so no
independent confirmation, causal recovery claim or architecture superiority claim.

## Software and evidence gates
All coding sessions use synthetic CPU data only. Read-only metadata verification
and original full-input A01 replay are allowed, but NO real completion calibration,
training or degraded evaluation. Real pilot then review then cohort is a separate
explicit operation. New source/config/packages requires fresh output identity.
Strict metadata/schema/path checks; refuse foreign outputs, changed identity,
partial/failed task retry and artifact corruption. Resume only verified completed
tasks and recheck historical input hashes. Scientific identity records reused
source plus new package, packages/config and historical manifests. Protect output
from symlink/path overlaps with source/data/historical roots.

Supervisor independently executes acceptance after each worker; failure/timeout
stops progression. A bounded repair, if needed, preserves the failed receipt and
states the defect. Workers provide handoffs with actual commands, outcomes and
limits. No silent retries or synthetic scientific claims.
