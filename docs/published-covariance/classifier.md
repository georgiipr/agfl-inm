# Frozen published classifier

`published_covariance.classifier.FrozenClassifier` loads one of the audited
official EEG-ATCNet run-1 checkpoints (A01–A09) with the pinned native
TensorFlow/Keras graph. Before loading, it checks the repository and revision in
`origin.json`, verifies the exact SHA-256 bytes of `models.py`,
`attention_models.py` and the Apache-2.0 license, then verifies the selected
checkpoint digest. Symlinks and paths escaping the supplied asset root are
rejected. The `.h5` extension is not used as an identity check.

The narrow native definitions in `published_covariance/atcnet_native.py` retain
the source ATCNet MHA graph, layer defaults, weight names/shapes, BatchNorm
epsilon, padding, pooling, activation, fusion and four-class softmax. Their
attribution and license are recorded in that module; the full pinned upstream
source and license remain in the immutable audit asset bundle. All nine loaded
graphs have 115,172 parameters and 195 weights, matching the independent audit.

The public call accepts finite floating values shaped `[B,22,1125]`, casts to
float32 as used by the checkpoint, inserts the source task axis, and returns
`[B,4]` probabilities. It does not append a mask or apply another softmax.
Inputs with an empty batch, wrong dimensions, non-floating dtype, or nonfinite
values fail. The model and all BatchNorm buffers are frozen; calls always pass
`training=False`, even when a caller supplies `training=True`, while gradients
with respect to the input remain available for the later covariance optimizer.
Constructing a Keras graph consumes TensorFlow's ordinary initializer RNG before
the audited weights replace every model weight. The wrapper does not reset or
change global RNG, device, threading or determinism settings.

The runtime used here is `.venv-published-covariance/bin/python` with
TensorFlow 2.15.1 and NumPy 1.26.4. Checks ran on CPU because no CUDA device is
available in the worker environment. Native-vs-wrapper synthetic checks covered
all nine subjects, including probabilities, labels, input gradients, repeated
inference under `training=True`, and unchanged model state. The supervisor's
fixed T-only array comparison remains a separate check; this session did not
run it, score E, or fit covariance. The checkpoint selection lineage remains
grade B as described in the checkpoint audit.
