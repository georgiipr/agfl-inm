# Session 05 integration review

## Scientific and reporting checks

- Completion is fitted once per task from the original normalized training raw
  trials. Covariance estimates the fixed channel second moment; Tucker uses the
  declared ranks, ridge, epochs, learning rate, batch size, and scoped CPU seed.
  No classifier parameters or validation labels enter either fit.
- For partial masks, observed values are selected with `torch.where` before
  completion arithmetic. The adapter carries the original Boolean mask into
  the historical classifier. It routes completed raw values through the
  existing front-end layers without zeroing imputed entries again. Full masks
  use the original backbone forward path, and zero completion has exact
  full-input logit equality in the synthetic CPU checks.
- Per-window masks reject all-missing windows. Hidden NaN/Inf invariance and
  exact observed-value preservation are covered by synthetic tests. The
  declared EEGNet whole-trial convolution remains in place; this completion
  is noncausal and does not recover a wholly missing window.
- Mask banks and ordered validation IDs are paired across all six cells. The
  verifier regenerates masks from the declaration, verifies mask digests,
  recomputes balanced accuracy and log loss from probabilities, and compares
  full-input probabilities across strategies. Reports average repeats, then
  seeds within participant, then participants equally. Negative participant
  effects are retained. The backbone interaction is reported as a
  difference-in-differences and is not described as synergy.
- Checkpoint selection is reused from full-input validation, so any resulting
  confidence intervals remain exploratory. Full-input bypass means this study
  cannot establish a clean-input accuracy gain.

## State and provenance checks

Each completed task now saves the complete primitive covariance and Tucker
state, including covariance second moment, fitted flag and sample count; Tucker
factors, fitted flag, fit-step count and channel seen-counts; and the numeric
settings that define both completers. `load_completion_state` loads with
`allow_pickle=False`, checks exact array names, scalar setting types, exact
study settings, tensor shapes and dtypes, finite values, fitted flags, Tucker
epoch count, factor unit norms, and valid covariance/count state. Its history
validator requires one correctly numbered row per fitted epoch with finite,
nonnegative diagnostics. Resume and reporting validate these values after
checksum checks, so recomputing a checksum around malformed numeric state does
not make that state acceptable.

Task output initialization rejects an output equal to or containing the
repository root, any overlap with data or historical input roots, and unowned
files. Historical replay verification
checks saved task, fit, checkpoint, prediction, split, data, normalization,
source and package identities before task preparation. Readiness verifies
all 27 task metadata records and matches their recorded AxxT.gdf hashes to the
current nine recording hashes. Historical inputs were read only. A01 replay
staging was temporary and removed after comparing original full-input
validation predictions.

## Evidence boundaries

Readiness consists of source review, the 36-test synthetic CPU suite, a
synthetic CLI smoke, the dependency-light plan and preflight, all 27 historical
metadata verifications, the nine recording byte-hash comparisons, and A01
original checkpoint replay. It is software and pilot readiness evidence. No
no completion factors were fitted on real EEG, no degraded inputs were
evaluated, and no new real accuracy result exists. Synthetic factors were fit
only as part of the software smoke. CUDA was not used or assessed. See
`validation/` for persisted command outputs and hashes.
