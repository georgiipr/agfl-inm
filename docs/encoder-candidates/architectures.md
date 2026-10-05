# Encoder candidate architectures

This document describes software in the fresh encoder-candidate study. It does
not report an experiment or predict which model will classify better. All raw
models receive normalized `[B,22,4,250]` inputs and Boolean availability masks
`[B,22,4]`; unavailable samples are selected to zero with `torch.where` before
filtering or learned mixing. Every model returns four logits.

## Spatial EEGNet reference

`SpatialEEGNet` wraps the current `EEGNetClassifier` with the study settings:
F1=8, depth multiplier 2, F2=16, temporal kernel 64, separable kernel 16,
pooling 4 then 8, dropout 0.5, flatten head, and mask conditioning. The EEGNet
front end changes `[B,22,4,250]` into a `[B,31,16]` feature sequence: temporal
filtering is followed by full-electrode spatial filters, then separable temporal
filtering and pooling. The reference classifier flattens those learned features,
appends the 88 original channel/window availability flags, and predicts four
classes. Because its spatial convolution combines electrodes, its feature
channels are learned activations rather than identifiable electrodes.

## Fixed spatial filter bank

`SpatialFilterBank` designs six 81-tap Hamming FIR band-pass filters with
`scipy.signal.firwin`, at 250 Hz, for 4–8, 8–12, 12–16, 16–20, 20–24, and
24–30 Hz. Coefficients are persistent frozen buffers. Each FIR runs with
same-length zero padding on each channel and each 250-sample window separately;
filter state never crosses a window boundary. There is no input instance
normalization before the FIR or variance calculation.

Per band, four learned spatial filters combine the 22-channel dimension inside
each window, yielding 24 latent components per window. Population variance is
calculated over each component's 250 temporal responses, followed by
`log(variance + 1e-6)`. The shape changes from `[B,22,4,250]` to ordered
`[B,4,6,4]` window/band/component features, flattened to 96 values. The model
appends the 88 original availability flags, applies dropout 0.5, and uses a
linear four-class head. The 24 component positions are learned latent mixtures,
not original electrode identities. Each spatial filter is clipped to max norm 1
over its 22 input channels.

The filter bank is a simplified window-local FBCNet-inspired adaptation. It is
not an external implementation or a reproduction of a published model.

## Compact spatial Transformer

`SpatialTransformer` initializes the same EEGNet modules as
`SpatialEEGNet(seed)` and retains only the convolution, normalization,
activation, pooling, and dropout layers that form its `[B,31,16]` temporal
sequence. The reference's flatten classifier is not retained in the
Transformer model. This makes the paired front-end tensors exactly identical
at initialization for a given seed, while leaving all retained front-end
parameters trainable.

A learned linear projection maps each position from 16 to 32 features. The
model adds a fixed sinusoidal code for positions 0–30, then applies one
batch-first Transformer encoder layer with four heads, width 64 in its GELU
feed-forward block, dropout 0.1, and pre-layer normalization. A final LayerNorm
is followed by the mean across 31 positions. The resulting 32 features are
concatenated with the original 88 Boolean channel/window availability flags
before the four-class linear head. There is no CLS token, learned position
embedding, or additional layer. `forward_features` returns the post-Transformer
sequence `[B,31,32]`; `position_encoding` is a persistent fixed buffer.

The default model has 11,092 trainable parameters; the fixed position buffer is
not counted. This is a complete-head comparison against the spatial EEGNet
reference: projection, temporal attention, positional codes, sequence pooling,
and the classifier all differ. It is not a pure test of temporal ordering, a
published EEG Conformer reproduction, or evidence of improved accuracy.
