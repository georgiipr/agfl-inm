# Covariance completion

The input trial is `[B,22,1000]`, viewed as `[B,22,4,250]` only to apply
per-window availability. The covariance acts over electrodes at each sample;
it does not compress time or introduce temporal factors.

## Training-only initialization

Flatten normalized full-channel **training** trials into sample rows
`Z ∈ R^(N_train × T)×22`. Channel normalization uses training mean and standard
deviation, so the second moment approximates centered channel covariance:

`Σ₀ = ZᵀZ / (N_train T)`.

Symmetrize in float64 and initialize from `chol(Σ₀ + 10⁻⁶ I)`. Frozen and learned
arms use the same initialization. No validation/test signal or label enters
this calculation. A calibration cache records normalization, the second moment,
training-trial count and its source/split identity.

## Positive-definite parameterization

Let θ contain the 253 lower-triangle coordinates of a 22×22 Cholesky matrix L.
Off-diagonal values are unconstrained; diagonal values are
`Lᵢᵢ = softplus(θᵢᵢ) + 10⁻⁶`. Then `Σ = LLᵀ` is positive definite.
The initial diagonal uses the inverse softplus, so construction preserves the
initial Cholesky matrix instead of silently transforming its scale.

Frozen θ is a buffer. Learned θ is the only parameter optimized during
completion calibration, and is frozen afterward. All solves use float64.

## Missing-channel inference

For observed electrodes O and missing electrodes H in one window:

`α = 0.001 mean(diag(Σ))`

`X̂_H = Σ_HO (Σ_OO + αI)⁻¹ X_O`.

The implementation solves the linear system rather than explicitly inverting
it. Equal masks share one solve within a batch. Values marked missing are
selected away before arithmetic; placeholders, NaNs or hidden reference EEG
cannot enter the solve. Preserve observed samples exactly and fill H only.
The full mask returns the original input without reshaping/completion. Both
backbones retain the original availability mask for their MHA key rules.

This is a regularized linear conditional estimate. Correlation is not causal
connectivity, and a completed sample is an estimate rather than a measurement.

## Supervised adaptation

After the original full-channel validation selects the backbone, hold all
backbone weights, BatchNorm buffers and dropout fixed. Draw train-reserved
22/16/11/6-channel masks from the same four outage families used in evaluation.
Construct observed inputs using the original raw-outage-aware filter.

With frozen classifier f and initial coordinates θ₀, optimize:

`J(θ) = weighted_CE(f(complete_θ(X_observed,M),M), y_train)`

`         + 0.1 mean_hidden((X̂ − X_train_full)²)`

`         + 0.0001 mean((θ − θ₀)²)`.

The reconstruction term uses artificially hidden **training** targets only.
Validation/test hidden targets are read separately for post-fit NRMSE reporting.
Wholly full batches skip updates, avoiding anchor-only changes on a bypass.

Use Adam, learning rate 0.001, batch size 32, no weight decay or scheduler,
gradient clipping at 1, and 100 fixed epochs. Retain the final epoch; neither
degraded validation nor test chooses completion weights. Save all training
masks, sample IDs, batch orders, losses and hashes. Runtime checks require
unchanged backbone tensors and exact unchanged full-input probe logits.

## Relationship to the research branch

The initialization, SPD parameterization, conditional solve and three-term
adaptation objective follow the inspected
[research implementation](https://github.com/georgiipr/agfl-inm/blob/e2e1fa893e4de50ea54ffa0e41d3624da902296d/inm/task_driven_completion/completion.py).
The current setup keeps its restored backbones, preprocessing and held-out test
protocol. It adds 11-channel and spatial training outages and uses fixed final
calibration weights instead of the branch's degraded-validation selection.
Consequently this is a new matched experiment, not a reproduction of that
branch's numerical results.
