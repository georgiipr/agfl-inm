# Four models with built-in MHA

The concrete classes are `EEGNet`, `EEGNetTensor`, `SignalTransformer` and
`SignalTransformerTensor`, in `eeg_models/models/`. Use `build_model(model_key,
metadata, model_options, window_samples=250, tensor_options=...)` to construct
one of these models. There is no attention argument, attention registry or
attention factory. MHA uses four heads directly in each backbone.
The fixed implementation in `_shared/layers.py` uses the original separate
linear Q/K/V and output projections, explicit scaled softmax scores, and an
independent seeded CPU RNG stream per attention layer. Constructing attention
does not advance the surrounding backbone's RNG. EEGNet retains the original
convolution/attention/classifier construction order.

## EEGNet

A full trial `[B,22,1000]` enters EEGNet's temporal convolution and BatchNorm.
The `[B,16,22,1000]` result gives 22 electrode tokens at each time step. Sensor
identity and built-in MHA mix those tokens, with a residual connection. The
result enters the full 22-channel depthwise spatial filter, ELU and pooling,
then the separable temporal convolution, pooling and flattened classifier.
All seven final pooled time bins remain ordered. There is no parallel classifier
branch, independent-window head or frozen channel-local encoder.

Defaults are temporal kernel 125, F1=16, depth multiplier 2, F2=32, pooling 8/16,
dropout 0.5, and max norms 1.0 for the spatial convolution and 0.25 for the
classifier. MHA operates before the depthwise spatial filter. Accuracy must be
measured under the declared training, artifact, filtering and selection protocol.

## Signal Transformer

A shared per-electrode temporal convolution (kernel 15), GELU, eight ordered
adaptive temporal bins and projection produce one 64-dimensional token per
original electrode. Two residual blocks each contain LayerNorm, fixed MHA and
an MLP. Final LayerNorm and learned signed spatial readout feed the classifier.
Temporal bin order remains in the tokenizer; attention mixes electrodes rather
than time positions. The tokenizer and every classifier layer are trained end
to end, independently of EEGNet.

## Masks and tensor variants

Every `forward(raw, mask)` accepts raw `[B,C,T]` signals and Boolean `[B,C,P]`
availability. `SignalInput` uses selection before any convolution or arithmetic
on hidden entries. Missing normalized samples become zero in baseline models,
which is the training-channel mean. The channel axis always has 22 original
sensor positions; it is never physically shortened or renumbered.

Tensor variants use Tucker-2 to complete only missing signal windows before the
same backbone. The signal tensor is `[B,22,4,250]`: channels and within-window
samples have training-fitted factors; the four window positions stay ordered.
Factors are frozen buffers. No classifier parameters are frozen. Full-channel
training/validation bypass completion exactly, preserving every signal sample.
Paired models use matched initialization and batch order; the tensor intervention
therefore targets inference under missing inputs, not full-channel compression.

Both variants retain the original mask for MHA keys. EEGNet excludes unavailable
electrodes at each time sample. Signal Transformer excludes an electrode key
only if it is absent throughout the trial; partially observed channels retain
their masked/completed temporal token. Inferred channels can still contribute
through queries, residual paths and the backbone's spatial readout. There is no
extra availability embedding or trainable representation adapter. Existing
sensor identities inside the architectures are retained.

Full-channel filtering uses the original recording-run bandpass before cutting
trials. Full EEGNet inputs take the original unmasked MHA route, including the
original checkpoint parameter names. For degraded inputs, raw missing intervals
are removed before filtering observed spans independently per electrode.
Outside-trial recording context remains observed. No filter crosses an outage,
native gap or run boundary. Temporal convolutions run across the entire
masked/completed trial; their neighbors are observations, zero placeholders or
estimates, never hidden reference samples. Train-only channel normalization is
shared by both backbones and variants.

See [tensor mathematics](tensor_math.md) and [experiment protocol](experiment.md).
