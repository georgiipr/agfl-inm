# EEGNet feature and attention adaptation

## Purpose and scope

The new experiment compares spatial attention on EEGNet-derived features with
and without a tensor representation under controlled electrode unavailability.
This is not a reproduction of standard EEGNet or the older AGFL project's
`spatial_fusion` architecture. Their full-channel spatial convolution cannot be
applied before simulated channel removal without mixing unavailable signals
into the remaining channels.

`inm/model.py` is the complete reviewable model implementation. The three
convolution blocks reproduce the channel-local path in the retained
`agfl/models/eegnet/backbone.py`. MHA and Performer are imported from
`agfl/attention/`. MHA is standard scaled dot-product attention; the retired
bias/gate branches have been removed. Performer retains its random-feature
implementation.

## Shared encoder

An input trial has 22 channels and 1,000 samples. Split each channel into four
nonoverlapping 250-sample windows **before** convolution. Apply shared temporal
convolution (16 filters, kernel 32), a grouped 1-by-1 expansion (depth multiplier
2), and separable temporal convolution (32 output filters, kernel 16). Pooling
factors 8 then 16 leave one sample per window and therefore 32 features. The
result is `X[B,22,4,32]`. There is no convolution across electrodes or across
window boundaries. The 1-by-1 expansion is not standard EEGNet's full-head
spatial convolution.

For a raw availability mask `M[B,22,4]`, `torch.where(M, raw, 0)` runs before
the first convolution; the output of each missing window is zeroed again.
This also blocks hidden NaNs, unlike multiplication by zero. BatchNorm uses
shared statistics while fitting the all-observed encoder, then frozen training
statistics during feature extraction. No validation/test BatchNorm updates
are permitted.

Fit the shared encoder with an MHA spatial classifier using training labels
only, select its epoch with validation data, and freeze it once per subject
and seed. Discard the pretraining classifier. All later attention and tensor
arms receive the identical frozen feature values. This is a deliberate
departure from the proposal's fixed spectral features: it preserves the
requested EEGNet feature family, but does not establish that the results
would hold for the proposal's spectral encoder. MHA-based pretraining is a
shared feature-source choice and should be disclosed because it may favor
MHA. It is not evidence that any tensor/attention method improves end-to-end
standard EEGNet.

## Spatial classification and masking

Each window is processed independently. Baseline tokens have shape
`[B*4,22,32]`. A tensor core has shape `[B,Rc,4,Rf]`, giving `Rc` tokens per
window; these are **latent spatial components, not anatomical regions or
electrodes**. Tensor completion reconstructs 22 electrode tokens instead.
Every classifier uses a 32-dimensional projection, learned node identities,
one residual attention/FFN block by default, layer normalization and a linear
four-class head. The default is four attention heads. No temporal attention
or time-position bias is used. Available windows are averaged at readout.

All attention families receive (1) the full 22-element observed-channel mask
for that window and (2) each electrode's observed flag, or the observed-channel
fraction for a latent core token. Baseline missing tokens use learned
placeholders and are excluded from the final spatial mean. They can still
participate in attention, with their unavailable status explicitly encoded.
This is a common **mask-conditioned fixed-token control**, not an exact
softmax key-padding mask. This convention applies to both MHA and Performer
and must be reported.

Tensor/mean completion pools both observed and imputed electrode tokens while
retaining original observation flags. Tensor/linear cores pool latent tokens.
Completely absent windows are rejected in every arm; each declared mask retains
at least six electrodes in every window. Those readout changes are part of each declared
representation, not hidden attention-specific changes.

## Controls and interpretation

- **Baseline:** measured features plus learned missing placeholders; no tensor.
- **Tensor core:** frozen training-only Tucker factors and masked core inference;
  attention over the latent spatial components.
- **Tensor completion:** the same factors reconstruct missing entries, while
  preserving observed values exactly; attention remains over electrodes.
- **Linear core:** supervised trainable channel/feature bottlenecks with the
  same core dimensions, without a reconstruction objective.
- **Mean completion:** fill missing channel-feature entries with the training
  mean, retain observed values and flags, and attend over electrodes.

The baseline and tensor core use both attention methods. Completion,
linear-core and mean-completion controls use MHA. Report parameter counts and
representation sizes; equal token/feature dimensions do not imply equal
parameter counts or training objectives. If normalization makes training means
nearly zero, report that rather than interpreting mean fill as a distinct
learned reconstruction method.

Classifier initialization is paired across attention methods within a
representation: common parameters are created before attention modules, and
the upstream attention factory isolates its random draws. Different
representations intentionally have different shapes and parameter budgets.
Neither accuracy improvement nor successful runtime has been established by
local execution; the implementation is prepared for cluster validation.
