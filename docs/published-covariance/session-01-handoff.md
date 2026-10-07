# Session 01 handoff

**Decision:** admit the complete official EEG-ATCNet `run-1` A01–A09 set as
structurally compatible, provenance grade B. Freeze this family before any
local classifier accuracy inspection. Do not select a different run by score.
The worker did not run a fit or evaluate E metrics.

The pinned source revision is `65162fb359ea46a2f62c885a9987247ab491ee7a`
(license Apache-2.0); the weights were uploaded in commit
`1ef6beb7ca6d60dcec146dadf9bc9299a15ac68b`. The JSON records each full asset
SHA-256. Direct HDF5 inspection found 195 finite float32 datasets / 115,172
elements in every subject file and matching tensor signatures, plus HDF5
metadata `backend=tensorflow`, `keras_version=2.4.0`, and 152 layer names. These
names/shapes match the pinned `ATCNet_` graph: five MHA windows, five temporal
heads, averaging and four-class softmax, input `[B,1,22,1125]`.

The native author-code recipe and line-level references are in
[checkpoint-audit.md](checkpoint-audit.md), with machine-readable details in
[checkpoint-audit.json](checkpoint-audit.json). In summary, the loader takes
22 channels in original order, 250 Hz input, direct stored trial start, and
samples `[trial+375:trial+1500]`; converts labels 1–4 to 0–3; retains all
artifact flags; applies no filter or reference in the BCI path; shuffles T
with seed 42; then fits each channel's 1,125 timepoint-specific StandardScaler
features over all T trials and applies those T statistics to E. Because this
normalization precedes the revised T 80/20 split, covariance validation trials
contribute to the historical scaler.

A01's official structured MAT file is recorded at
`.session-runs/published-checkpoint-covariance/assets/recordings/A01T-origin.json`.
Supervisor's read-only MAT/GDF receipt compares all 288 included windows and
supports first-22-channel locality for A01. The native loader is one sample
(4 ms) after the zero-based GDF cue marker; preserve that source behavior.
The receipt's numeric difference is diagnostic, not an acceptance threshold.
Session 02 must declare its comparison tolerance first and repeat input/source,
channel, cue, label, trial-exclusion and normalization checks for A02–A09,
including E arrays. Never treat a label-only competition E MAT as the
structured EEG input expected by ATCNet.

Grade B is deliberate. Repository chronology and the revised log support the
T-validation recipe, but there is no per-checkpoint training receipt tying the
uploaded H5 bytes to that log. The same repository retains an older routine that
uses E for epoch checkpointing and best-run selection. Later reports must
preserve this uncertainty and the project's unknown prior E exposure. The
revised script uses a single run, checkpoints by T validation loss and evaluates
E afterwards, but those facts alone do not prove which training invocation made
each current tensor.

Proposed runtime is in
[dependencies-proposed.txt](dependencies-proposed.txt): CPython 3.11,
TensorFlow 2.15.1 / bundled `tf.keras`, NumPy 1.26.4, SciPy 1.11.4,
scikit-learn 1.3.2 and h5py 3.10.0. H5's Keras 2.4.0 attribute does not
identify TensorFlow's exact version. This choice uses the pinned native source
and avoids a framework port, but remains unverified until the supervisor loads
all nine files, performs a native synthetic forward/backward, confirms frozen
inference behavior and records GPU/CUDA/package details. The coding worker did
not install packages or touch `.venv`.

## Evidence links

- [Pinned EEG-ATCNet source revision](https://github.com/Altaheri/EEG-ATCNet/tree/65162fb359ea46a2f62c885a9987247ab491ee7a), [Apache-2.0 license](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/LICENSE)
- [ATCNet graph and input](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/models.py#L34-L117), [MHA graph](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/attention_models.py#L60-L93)
- [Legacy E-guided train/test path](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/main_TrainTest.py#L118-L177)
- [MAT trial extraction](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/preprocess.py#L85-L148), [shuffle and scaler](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/preprocess.py#L308-L365), [revised T split](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/main_TrainValTest.py#L120-L164)
- [Official BCI competition download page](https://www.bbci.de/competition/iv/download/)
