# Native data and static-mask protocol

## Recording adapter

`published_covariance.data.load_native_recording(path, subject, session)` reads
only the author's structured `AxxT.mat` / `AxxE.mat` format, where `data` is a
row of run structs. A label-only competition MAT file is rejected. The caller
must choose `T` or `E` explicitly, and the file stem must match that identity;
there is no implicit E lookup in the T calibration path.

The adapter follows the pinned EEG-ATCNet loader at revision
`65162fb359ea46a2f62c885a9987247ab491ee7a`: 250 Hz, first 22 data columns in
their original order, and the time-first interval
`X[start:start+1750, :22][375:1500, :].T`, yielding `[22,1125]` per trial.
The complete 1,750-sample outer interval must exist, even though the final
crop ends at `start+1500`; a short recording fails closed and no annotated
trial is silently omitted. Thus the effective source indices are
`[int(a_trial)+375:int(a_trial)+1500]`. The stored MATLAB integer is used
directly; do not subtract one. This is four and a half seconds, nominally 1.5–6
seconds after the stored trial start. It preserves the author's one-sample
(4 ms) offset relative to a zero-based GDF cue marker.

No filtering, rereferencing, resampling or unit conversion is introduced.
Values remain in the source's microvolt units. Source channels map to the frozen
order `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1,
CPz, CP2, CP4, P1, Pz, P2, POz`. Source labels 1–4 map to class IDs 0–3 in
`left, right, feet, tongue` order. Every annotated trial is retained, including
nonzero artifact flags (`all_trials=True`).

`NativeRecording` returns values and labels in original run/within-run order,
1-based `run_ids`, Boolean `artifacts`, unique IDs of the form
`A01:T:r04:t012`, and metadata. The ID preserves subject, session, source run,
and within-run cue order; split indices therefore point back to the untouched
chronological native arrays. Metadata records the source file SHA-256, input
recipe, channel/class mapping, artifact policy, per-run trial counts, included
artifact count and an empty excluded-trial list (native bounds violations are
errors). It does not drop artifact-flagged trials. Labels, starts, sampling
rates and artifact flags must be finite integral scalars before conversion.
Raw extraction performs no arithmetic and leaves nonfinite sensor values for
`normalize`, which selects retained electrodes first and rejects nonfinite
observed values. Fit data remains subject to the same finite-value checks.

## Native normalization and covariance split

`fit_native_normalization(values)` reproduces the author's `shuffle(...,
random_state=42)` order and fits one `sklearn.preprocessing.StandardScaler`
per channel to a `[T trials,1125 time points]` matrix. Each scaler therefore has
1,125 means and scales. `save_normalization` records exact arrays, a validated
64-character T source hash, subject, ordered T-only IDs for that subject and
their digest. It creates the destination exclusively, so a second writer cannot
replace the provenance record. `load_normalization` verifies schema, subject,
source hash, trial-ID format and digest, plus finite numeric arrays with
positive scales; callers can additionally require a matching subject and source
hash. This is a replay of the
published classifier's T-fitted transform, which historically saw all T trials
before the new covariance split. It is distinct from covariance calibration:
the covariance fit receives only the split's training indices. No new
normalization is fitted on E.

`normalize(values, mean, scale, mask)` accepts a Boolean `[N,22]` whole-trial
mask. It applies `where` before subtracting or dividing, rejects nonfinite
observed data and empty masks, and emits zero at hidden normalized entries.
Consequently a hidden NaN or infinity cannot contaminate retained channels.
With a full mask, same-runtime output matches the native transform bitwise.

`stratified_split(labels, seed)` operates on original chronological indices.
It uses `Generator(PCG64(seed))`, visits classes in 0,1,2,3 order, permutes each
class's trial indices, allocates the first `max(1,floor(.2*n_class))` to
validation, and returns sorted train/validation indices in source order. It
fails if any class would have no training examples. The split seed is a
covariance seed, not a pretrained-checkpoint seed.

## Whole-trial static masks

`masks.static_masks(subject, seed, trial_ids, partition, retained, repeat=0,
epoch=0)` produces Boolean `[N,22]` masks for retained counts 22, 16 or 6.
There is no time/window axis because this study removes an electrode for the
complete trial and all 1,125 samples. Full 22 is all true. Partition names are
`train`, `validation`, and `E`.

The key is compact JSON array encoding of
`["static-subset-sha256-pcg64-v1", namespace, subject, seed, original_trial_id,
partition, retained, repeat, epoch]`; SHA-256's first 128 bits seed NumPy
PCG64. No Python `hash()` or labels/features are used. Each nonfull candidate
subset is assigned to one stable partition pool by interpreting the 22-channel
bitset and mapping the first 64 bits of
`SHA256("static-subset-sha256-pcg64-v1:partition:" + decimal_bitset) modulo 3`
to train/validation/E. Uniform `choice(22, retained, replace=False)` draws are
rejected until one belongs to the requested pool, making sampling uniform
within that pool. This guarantees no exact subset can cross the train,
validation, and E pools. Identical keys yield paired masks across all methods;
changing any key component changes the deterministic stream. Namespace defaults
to `published-checkpoint-covariance-v1` and can be versioned explicitly.

## Checks and limits

The declared native comparison used bitwise equality for MAT extraction and
same-runtime normalization, `1e-8` microvolt absolute / `1e-10` relative
tolerance for GDF samples, and exact channel, label and trial alignment. The
A01 T structured MAT output and labels compare bitwise with the pinned author's
`load_BCI2a_data(..., training=True, all_trials=True)` and contain 288 trials.
The supervisor's cohort alignment receipt
`.session-runs/published-checkpoint-covariance/02/native-recording-alignment.json`
records all 18 A01–A09 T/E recordings: 288 trials each, zero exclusions,
channel/time equality against the original GDF at the declared tolerance, T
labels matched to GDF events, and E labels matched to the official label-only
MAT files. The observed maximum sample error is about `1.43e-14` microvolts.
The repair session ran synthetic data/persistence checks plus the independent
A01 native-array check; it did not load E through the adapter, run a classifier,
fit covariance, or compute any outcome metric.
