# Tucker-2 signal completion: equations and implementation

The intervention is implemented in
[`eeg_models/models/_shared/tensor.py`](../eeg_models/models/_shared/tensor.py). It uses
only PyTorch, already required by the original codebase. “Tensor” and
“hypermatrix” refer to the same higher-order array in this project. Merely
storing signals in such an array is not the intervention: the fitted multilinear
factors and inference from the available entries are the intervention.

## Input and observation mask

For each recording, let

\[
X\in\mathbb R^{C\times P\times F},\qquad
M\in\{0,1\}^{C\times P}.
\]

There are \(C=22\) named EEG electrode positions, \(P\) disjoint temporal windows,
and \(F=250\) normalized time samples per electrode/window. The known electrode universe is
retained when channels disappear. The mask is broadcast over features. Only
\(M_{cp}=1\) entries are inputs; the other array positions may contain arbitrary
placeholders, including NaN. Observed samples must be finite.

**Implemented representation:** normalize EEG using train-only channel
statistics, then reshape each full trial to `[B,22,4,250]`. The feature-mode
coordinate in the mathematics below is a within-window sample, not a frozen
learned feature or spectral coefficient. Completion preserves the raw model
interface `[B,22,1000]` after reshaping. EEGNet and Signal Transformer both
train end to end with built-in MHA. The experiment tests multilinear signal
completion within these classifiers.

The API selects observed entries using `torch.where` **before arithmetic**.
Multiplying hidden values by zero would be unsafe because `0 * NaN` is NaN.
Every recording/window must retain at least one channel. The tensor module
accepts any positive window count and any observed channel count from 1 to 22.

## Shared representation and exact core solve

Use training-fitted factors and an uncompressed temporal mode:

\[
U\in\mathbb R^{C\times R_C},\quad
V\in\mathbb R^{F\times R_F},\quad
G\in\mathbb R^{R_C\times P\times R_F},\quad
\widehat X_p=UG_pV^\top.
\]

Columns of \(U\) and \(V\) have unit Euclidean norm. The full preset uses
channel rank 4 and sample-mode rank 16; neither rank can exceed its
corresponding input dimension. Fixed ranks are declared in the experiment
configuration, rather than selected on held-out test accuracy.

For window \(p\), let \(O_p=\{c:M_{cp}=1\}\), \(n_p=|O_p|\), and
\(D_p=n_pF\). For fixed factors, infer

\[
G_p^*=\arg\min_G
\underbrace{\frac{\|X_{O_p,p,:}-U_{O_p,:}GV^\top\|_F^2}{D_p}
+\lambda_G\|G\|_F^2}_{\ell_p(G;U,V,X,M)},\qquad\lambda_G>0.
\]

The normalization is over **observed scalar features**. Consequently the normal
equation contains \(D_p\lambda_G\), not merely \(\lambda_G\):

\[
A_pG_pB+D_p\lambda_GG_p=C_p,
\quad A_p=U_{O_p,:}^\top U_{O_p,:},\quad B=V^\top V,
\quad C_p=U_{O_p,:}^\top X_{O_p,p,:}V.
\]

The implementation avoids a large Kronecker design. Compute the small symmetric
eigendecompositions \(A_p=Q_p\operatorname{diag}(a_p)Q_p^\top\) and
\(B=R\operatorname{diag}(b)R^\top\), then

\[
H_p=Q_p^\top C_pR,
\qquad
(\widetilde G_p)_{ij}
=\frac{(H_p)_{ij}}{(a_p)_ib_j+D_p\lambda_G},
\qquad G_p=Q_p\widetilde G_pR^\top.
\]

Gram matrices are symmetrized, roundoff-negative eigenvalues are clipped to
zero, and solves use float64. Positive ridge makes every denominator positive,
including when the number of observed channels is below the channel rank. Core
inference uses no gradient through this solve, no labels, no future-window
smoothing, and no updates to the shared factors. Its output is returned in the
input feature dtype.

## Fitting factors on training data only

For the current subject and seed, use only that subject's designated training
trials and training masks. The protocol trains nine subjects individually, using
within-subject partitions rather than subject-disjoint folds. The supplied
experiment fits one shared factor pair from **full-channel training signal
windows** and uses it in both tensor backbones. No labels, validation or test
samples enter factor fitting. All 22 sensors are observed during
training/calibration; availability loss is evaluated afterward. Fitting factors
does not freeze either supervised backbone. Non-tensor models share the same
training-only normalization and use zero for missing normalized samples.

Fit \(U,V\) by the constrained training-only objective

\[
\min_{U,V,\{G_{np}\}}\frac{1}{N_{\rm train}P}
\sum_{n\in D_{\rm train}}\sum_{p=1}^{P}\ell_p(G_{np};U,V,X_n,M_n),
\quad\|U_{:r}\|_2=\|V_{:s}\|_2=1.
\]

`fit` initializes normalized random factors using the caller's seeded PyTorch
generator. Each minibatch alternates an exact detached core solve with an Adam
update of the factors while holding those cores fixed. Factor columns are
projected back to unit norm after each update. This is projected alternating
optimization, **not** an assertion of a global optimum or monotone convergence.
After every epoch, fresh cores are solved to report the post-update training
objective, observed-entry MSE, and ridge penalty. Validation and test data do
not enter initialization, these losses, or factor updates.

The training mask participates in both the solves and the fitting loss. A
channel never observed anywhere in the factor-fitting training inputs causes an
explicit error: a free factor row for it is not identifiable from those data.
Once fitted, factors are PyTorch **buffers**, not model parameters. Supervised
optimizers therefore cannot silently fine-tune them. Fitted factors, fitted-
state flag, fit-epoch count, and training observation counts are in the state
dictionary; the surrounding experiment must also save the rank/ridge
configuration.

## Completion in the four models

`encode(X,M)` is the internal conditional core solve. `reconstruct(X,M)` returns
the factor estimate everywhere. The model intervention uses only
`complete(X,M)`:

\[
X^{\rm completed}_{cpf}=
\begin{cases}
X_{cpf},&M_{cp}=1,\\
(UG_pV^\top)_{cf},&M_{cp}=0.
\end{cases}
\]

Measured samples are preserved exactly; estimates fill only missing windows.
Completion occurs before the entire backbone and its built-in MHA. The original
mask continues to control the attention keys, distinguishing available channels
from entirely missing ones. The reconstructed signal keeps all sensor positions
and ordered windows. There is no separate learned core projection, availability
embedding, interchangeable attention head or window-averaged classifier.

`SignalInput` bypasses the factor solve when every channel/window is observed.
Thus the tensor intervention changes no signal sample in full-channel training
or validation and adds no trainable classifier parameters. Matched
initialization and batch order make full-input baseline/tensor behavior a
controlled comparison. Whether tensor completion helps on lost channels must be
measured experimentally.

The mathematical API is:

```python
tucker = Tucker2(
    channels=22, features=250, rank_channels=4, rank_features=16,
    ridge=1e-3, fit_epochs=30, fit_lr=1e-2, fit_batch_size=64,
).to(device)
history = tucker.fit(training_signal_windows, full_training_mask)
completed = tucker.complete(observed_signal_windows, availability_mask)
```

Signal windows have shape `[B,C,P,F]`, Boolean masks `[B,C,P]`, and internal
cores `[B,Rc,P,Rf]`. Restore factors using the same constructor rank/ridge
settings. Factor state, training observation counts, normalization, constructor
options, selected classifier weights and source/split identities are archived
together.

## Interpretation limits

Positive ridge guarantees a unique **conditional core solution** for fixed
factors. It does not guarantee accurate reconstruction, label information
retention, accuracy improvement, or statistical identifiability of the learned
factors. Tucker factors admit rotations/sign ambiguities; sharing frozen factors
keeps component coordinates consistent between training and evaluation.

Severe or spatially concentrated channel losses may remove information that no
factor model can recover. Static 6-of-22 availability removes 72.73% of
channels, the integer-channel approximation to 75% loss; the exact retained
count must be reported. Balanced accuracy, macro-F1, paired robustness changes,
and observed full-input degradation remain the decisive classification outcomes.
Hidden-sample reconstruction error is secondary and must be computed from
evaluation targets only after inference, never used as input or a fitting signal
on held-out data.
