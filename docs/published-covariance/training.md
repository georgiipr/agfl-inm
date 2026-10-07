# Covariance completion and training

`published_covariance.completion` implements zero fill and conditional Gaussian
completion for native `[B,22,T]` arrays. The covariance equations follow the
study contract and the earlier project's covariance implementation in
`inm/task_driven_completion/completion.py`, adapted as a standalone TensorFlow
implementation for the published classifier. The module does not import or
modify that earlier package. This is an implementation description, not evidence
of an accuracy benefit.

For each covariance training split, training-only normalized samples form the
uncentered second moment

\[
S = \frac{1}{N T}\sum_{n,t}x_{n,t}x_{n,t}^{\mathsf T}.
\]

The initialized covariance is `S + 1e-6 I`. Its Cholesky factor uses 253
float64 lower-triangle parameters; diagonal entries are `softplus(free) + 1e-6`.
The fixed and learned completers have identical initial parameters. The
conditional mean uses `rho = 0.001 * mean(diag(Sigma))` and solves the observed
principal system in float64. TensorFlow uses a masked 22-by-22 block system:
unobserved rows receive an identity block, and observed rows receive the
specified ridge. This is algebraically the same observed-only solve and permits
different masks in one batch without an explicit inverse.

Masks are selected before any arithmetic. Hidden entries are replaced with zero
before casting or solving, observed values must be finite, and empty masks fail.
After completion, observed entries are selected directly from the original input
tensor, preserving their values exactly. Full input returns unchanged. The
solver supports any positive native time length; the published classifier itself
still requires 1,125 samples.

`train_covariance` initializes only from its training arrays and rejects IDs that
do not identify the requested participant's T trials. Every training epoch
shuffles with a local PCG64 generator derived from the participant and seed.
Each trial independently draws 22, 16, or 6 retained electrodes with equal
probability. It uses the existing deterministic `static_masks` helper for
partition-disjoint subsets. Validation uses the same ten fixed conditions at
every epoch: five repeats each for 16 and 6 retained channels.

The frozen classifier is called in inference mode while gradients pass through
its input. The covariance alone is optimized with

\[
\operatorname{CE}(p,y)+0.1\,\operatorname{MSE}_{hidden}
+10^{-4}\,\operatorname{mean}((free-initial)^2).
\]

The native four-class probabilities go directly to categorical cross-entropy;
there is no second softmax. Adam uses learning rate `0.001`, betas `(0.9,0.999)`,
epsilon `1e-8`, batch size 32, and global gradient clipping at 1. There is no
schedule or weight decay. The code records per-epoch loss terms, validation
balanced accuracy and log loss, gradient norm, and distance from initialization.

Epoch zero is scored before optimization and remains eligible. The selected
state maximizes mean balanced accuracy across the ten validation conditions,
then minimizes mean log loss, then prefers the earlier epoch. Training stops
after at least the declared minimum epoch when patience expires, or at the
maximum. The exact selected 253-value state is restored after training. Synthetic
tests may override the epoch budget; real use requires the fixed 100/10/20
maximum/minimum/patience values. Neither E data nor E labels are accepted by this
API.

Synthetic validation establishes numerical equivalence and implementation
behavior only. It does not establish that covariance training improves
published-classifier robustness.
