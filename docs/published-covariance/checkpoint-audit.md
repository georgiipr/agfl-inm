# Published ATCNet checkpoint audit

Audit frozen 2026-10-06 against the complete official [EEG-ATCNet
repository](https://github.com/Altaheri/EEG-ATCNet/tree/65162fb359ea46a2f62c885a9987247ab491ee7a)
at `65162fb359ea46a2f62c885a9987247ab491ee7a`. The exact run-1 HDF5 files
entered repository history together in commit
[`1ef6beb7ca6d60dcec146dadf9bc9299a15ac68b`](https://github.com/Altaheri/EEG-ATCNet/commit/1ef6beb7ca6d60dcec146dadf9bc9299a15ac68b)
on 2024-01-31. The repository is licensed under
[Apache-2.0](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/LICENSE).
The read-only source and asset hashes are preserved in
`.session-runs/published-checkpoint-covariance/assets/origin.json`.

## Admission decision

**Admit the complete run-1 set provisionally as the one compatible checkpoint
family, with provenance grade B.** All nine files load as finite float32 weight
tensors with the same architecture signature (195 datasets and 115,172 scalar
elements per subject). The HDF5 metadata says `backend=tensorflow` and
`keras_version=2.4.0`. Each file has 152 layer names, including the temporal
convolutional stem, five layer-normalization / multi-head-attention windows,
five temporal heads, their average and four-class softmax. Those names and
weight shapes match `models.ATCNet_` and `attention_models.mha_block` at the
weight-upload commit. The model input layout is `[B,1,22,1125]` (task axis, electrode axis, time); the graph permutes it internally before convolution. No weights are converted or modified. This is source and tensor structure evidence, not yet a successful native framework load or forward pass.

The exact run history remains uncertain, so the cohort cannot be described as
having proven T-only checkpoint selection. The history contains both the older
`main_TrainTest.py` path, which validates/checkpoints and chooses runs using E,
and the revised `main_TrainValTest.py` path, which splits T, checkpoints on T
validation loss, and uses one run. The prior results directory and weights were
deleted on 2023-12-31; a revised training log and run-1 manifest were uploaded
that day, and the current run-1 HDF5 files were uploaded on 2024-01-31. The
log's format matches the revised script and reports one run. This supports the
revised T-only selection path, but no per-checkpoint training receipt or
weight-to-log digest proves the association. The current `best models.txt`
lists run 1 for each participant. Do not choose or replace any checkpoint based
on this study's accuracy. Grade B records this unresolved selection lineage;
known or unknown E-guided selection must remain disclosed in later reports.

The model and preprocessing recipe are resolved at the author-code input
boundary. The actual recording arrays still require the session-02 MAT-origin
and GDF-alignment gate. The source expects the author's structured
`AxxT.mat`/`AxxE.mat` `data` structure; it is not interchangeable with a
label-only competition `AxxE.mat`. No recording file or conversion script was
present in the checkpoint asset bundle when this audit was written. The
supervisor has since acquired the author's documented A01T structured MAT
input (URL and digest are in
`.session-runs/published-checkpoint-covariance/assets/recordings/A01T-origin.json`).
Read-only inspection confirms nine runs, 250 Hz, three calibration-only runs,
then six runs of 48 balanced labeled trials. The first trial index is 251. A
supervisor diagnostic compared all 288 retained native MAT windows to GDF and
found the source slice `[trial+375:trial+1500, :22]` aligns with GDF at
`[marker+376:marker+1501, :22]`, with maximum absolute difference
`1.42e-14` microvolts. The native code therefore starts one sample after the
zero-based GDF marker; preserve this 4 ms offset. This was a diagnostic, not a
predeclared acceptance tolerance. Session 02 must declare its tolerance before
its own parity checks. The same receipt reports offset-zero and offset-minus-one
errors of 84.326 and 71.338 microvolts. Verify the remaining participants
separately before freezing data identity.

## Reconstructed model configuration

At the pinned source, `main_TrainTest.getModel` constructs `ATCNet_` with four
classes, 22 channels, 1,125 samples, five sliding windows, MHA, EEGNet stem
`F1=16`, depth multiplier 2, kernel length 64, pool 7, dropout 0.3, two TCN
blocks of depth 2 with kernel 4 / 32 filters / ELU / dropout 0.3, and average
fusion of five four-class heads followed by softmax. The architecture source is
[`models.py` lines 34–117](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/models.py#L34-L117),
the MHA details are in
[`attention_models.py` lines 60–93](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/attention_models.py#L60-L93),
and the BCI IV-2a `getModel` parameters are in
[`main_TrainTest.py` lines 291–321](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/main_TrainTest.py#L291-L321).

The model has no appended mask input. The published input stays
`[B,1,22,1125]`; completion must return normalized values in that exact layout.
Inference must set the Keras model to inference behavior (BatchNorm moving
statistics and dropout disabled) while retaining gradients to the input for
covariance training.

## Native input recipe from the pinned author source

`preprocess.load_BCI2a_data` opens one structured MAT recording per session and
participant, reads the `data` entries' continuous matrix, event sample indices,
labels and artifact flags, takes the first 22 columns, slices a 1,750-sample
trial beginning at the native `a_trial` index, then keeps `[375:1500]`. The
script hardcodes 250 Hz, so the classifier sees the 4.5-second interval from
1.5 through 6.0 seconds after that trial index: 22 channels by 1,125 samples.
Class labels are converted from 1–4 to 0–3. `all_trials=True` is the default,
so this source does not exclude artifact-flagged trials. The relevant code is
[`preprocess.py` lines 85–148](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/preprocess.py#L85-L148).
The input/model dimensions and BCI dataset declaration are
[`main_TrainTest.py` lines 353–393](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/main_TrainTest.py#L353-L393).

There is no bandpass, notch, rereference, resampling, or artifact rejection in
this BCI path after the `data` struct is loaded. Preserve its existing sample
values; do not add the legacy study's filter or reference. The first 22 channel
identities must be retained in their original order. The source applies no cross-channel operation before standardization. The
A01 native-to-GDF diagnostic supports the first-22-channel mapping and shows no
upstream spatial-mixing discrepancy for that participant; repeat this audit for
the other eight participants before masked evaluation.

`get_data` first deterministically shuffles the T trials and labels together
with scikit-learn `shuffle(random_state=42)`. It reshapes to `[trial,1,channel,
time]`, then fits a separate `StandardScaler` for each channel on the matrix
`X_train[:,0,j,:]` of shape `[T trials,1125 time features]`. This means there
are 1,125 independently fitted means/scales per channel, estimated across T
trials, not one scalar per channel and not a statistic shared across channels.
It applies those T-fitted per-timepoint transforms to both T and E. In the
revised recipe the T scaler is fitted before the subsequent 80/20 T split, so
it has seen the later covariance-validation trials; disclose this historical
exposure. There is no E-fitted transform. These operations are channel-local,
but the MAT producer's upstream locality remains to be verified. See
[`preprocess.py` lines 308–365](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/preprocess.py#L308-L365)
and the revised split in
[`main_TrainValTest.py` lines 120–164](https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/main_TrainValTest.py#L120-L164).

Because every channel/timepoint is standardized using statistics from that
same channel/timepoint, a consistent affine recording-unit change is removed
by the scaler. This does not establish electrode mapping, reference, filter,
or cue alignment; those require actual array verification. The official BCI IV
download page describes the competition dataset at
[bbci.de](https://www.bbci.de/competition/iv/download/). Do not pass the
competition's label-only E MAT as if it were the author's structured EEG MAT.

## Provenance and unresolved facts

| Subject | Checkpoint SHA-256 | T training | Selection lineage | Normalization | Grade | Remaining fact |
|---|---|---|---|---|---|---|
| A01 | `15a6a6cd7454d4d5b1b59adfe1d9636140fb0bd5a5e0f42adafadb4a36e9b1fd` | Session T | Revised T-val path supported; exact H5-to-log lineage unproven | Per-channel/per-timepoint T scaler, fit before T split | B | Verify remaining MAT/GDF alignment and resolve checkpoint receipt |
| A02 | `e9a26fca0e07a82b1374ffbdeedf3bf8d16ba67a39603d39160613a54590171d` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A03 | `acec4e58ac81d0df72c89e1910aca444cda11a8c76465780b709c318925c2e9b` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A04 | `87a3e5973b35d6e81500f6d134669a6dda16eeca41e24a90da7cbc757f457b92` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A05 | `685da44928c8c0d57dcd9247016b0a48f09af02e041a923596ff5c5077c1aeee` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A06 | `ddfffffa081e197b4e5117c24a186a056a74a358cc4f81f436864ae34b16be3d` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A07 | `a9cb9da09f0b52f331fdcdd47899a7eec3affa4d4a3cfd674e2e8c4ac85f0866` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A08 | `86ba692320a1d057b116b9a88f9b1de966ae7f93487bfc05d097fd87e18363c7` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |
| A09 | `46bc137b2b6e34e6ecdc092099918b2c2fd322830786bdbfc93d6f0ea7c97ba8` | Session T | Same cohort run-1 history; no subject sidecar | Per-channel/per-timepoint T scaler, fit before T split | B | Same; no subject-specific sidecar |

`results/best models.txt` selects run 1 for each subject. The complete set was
admitted by the fixed run-1 path declared before this audit, not by comparing
scores. The H5 files have identical tensor-shape/dtype signatures; weights
remain subject-specific. The precise older-versus-revised training invocation
and any external run selection are unresolved. The source's old path explicitly
uses E both for epoch checkpointing and best-run selection, so the grade must
remain B unless a weight-specific receipt resolves that possibility. The
project's prior exposure to E outcomes also remains unknown as specified by the
contract.

## Runtime recommendation

Use the pinned author TensorFlow/Keras source without a framework port, in an
isolated CPython 3.11 runtime with `tensorflow==2.15.1`, `numpy==1.26.4`,
`scipy==1.11.4`, `scikit-learn==1.3.2`, and `h5py==3.10.0`. The saved H5
metadata records Keras 2.4.0; it does not identify the TensorFlow release. This
proposed TensorFlow 2.15.1 runtime retains the legacy `tf.keras` API and
supports this Python version, but H5 loading, graph creation, forward behavior
and input-gradient parity remain unverified. The author code also imports
Matplotlib and `python-dateutil`; its BCI MAT path does not need MNE's EDF
reader. Do not install into `.venv`. The supervisor must create the isolated
runtime, record its resolved lock and GPU/CUDA details, and complete the
native-checkpoint smoke/parity gate before real fitting. If it fails, diagnose
the runtime before considering a port.
