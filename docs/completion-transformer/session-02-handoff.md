# Session 02 handoff: completion and frozen replay adapters

Completed and acceptance-repaired 2026-10-05. Added training-fitted completion
controls and layer-level routes for the existing spatial EEGNet and spatial
Transformer candidate models. Adapter checks live in the required
`test_adapters.py` module. Changes are limited to `inm/completion_transformer/`,
`tests/completion_transformer/`, and this handoff.

The package initializer keeps protocol exports eager and standard-library-only,
while lazily resolving adapter/completer exports. This preserves the planning
import contract without removing the convenient top-level numerical APIs.

## APIs

- `CovarianceCompleter(channels=22, ridge_scale=0.001).fit(train_x)` estimates
  the full-channel second moment from normalized training samples only. Its
  `complete(x, mask)` performs conditional ridge prediction independently for
  each window and observed-channel pattern, preserving observed samples.
- `TuckerCompleter(...)` constructs the existing `inm.tensor_attention.Tucker2`
  with the fixed rank, ridge, and fit settings. `fit(train_x, task_seed)` fits
  only the supplied training tensor under CPU seed `task_seed + 810001` and
  restores Python, NumPy, and Torch RNG states. Its model state dict can be
  saved/reloaded; factors remain Tucker buffers. `complete(x, mask)` delegates
  to frozen observed-entry inference.
- `CompletionAdapter(backbone, strategy, completer=None)` accepts `zero`,
  `covariance`, or `tucker`. For partial masks it routes selected or completed
  values directly through the existing EEGNet front-end layers and retains the
  original Boolean flags at the classifier. Fully observed input calls the
  original backbone directly. The adapter starts in eval mode, disables
  gradients, and rejects inference if the adapter, backbone, or completer has
  since entered training mode.
- Package exports are `CompletionAdapter`, `CovarianceCompleter`,
  `TuckerCompleter`, and `preserve_observed` from
  `inm.completion_transformer`.

## Verification

Executed from the project root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/completion_transformer -v
```

All 17 tests passed, comprising six protocol cases, six completion cases, and
five cases in `test_adapters.py`. The protocol cases include a subprocess check
that importing `inm.completion_transformer.protocol` leaves `torch`, `numpy`,
and `scipy` absent from `sys.modules`. Checks include explicit covariance
conditional-ridge and Tucker core-solve matrix oracles, with Tucker penalty
`n_observed * features * ridge`; observed-value preservation; hidden NaN/Inf
invariance; all-missing rejection; Tucker state reload and repeatability across
ambient RNG states; RNG restoration after a forced fit exception; original mask
flags reaching each classifier; completion affecting predictions; and frozen
classifier and BatchNorm state. Zero-fill logits match both original forwards
exactly. For every strategy, all-full input takes the original forward route;
the paired Transformer comparison allows `rtol=1e-6, atol=1e-7` for a measured
sub-ulp CPU variation (about `6e-8`) between repeated original Transformer
calls.

`compileall` and `git diff --check` passed. The only runtime notice was
PyTorch's existing convolution warning for `padding='same'` with an even
kernel. No tests failed in the final run.

## Limits and remaining work

These are synthetic CPU checks only. No historical checkpoint or split was
opened, no cohort data or validation metrics were evaluated, and no study
output was written. Session 01's metadata inventory is not a replay
compatibility verdict; the planned replay must still verify the saved fit,
constructor, selected epoch, normalization, split/data IDs, predictions, and
source/package identity before reading numerical evidence or running completion.
The fixed experiment uses 22 channels, four windows, and 250 samples per
window. Completion remains a noncausal per-window estimate and does not infer
a window with no observed electrode.
