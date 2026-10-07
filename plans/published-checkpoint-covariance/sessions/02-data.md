# Session 02: native data and static loss

Read contract, acceptance and verified session01 audit. Default20minutes;
one fresh worker, checks<=60seconds. No fitting or E metrics.

Write scope: `published_covariance/data.py`, `published_covariance/masks.py`,
corresponding tests under `tests/published_covariance/`,
`docs/published-covariance/data.md`, `session-02-handoff.md`.
Minimal new package initialization is allowed. Plans/assets are read-only.

Implement the audited native input recipe and original-ID provenance, official
labels, channel/class/unit mapping, inclusion/exclusion rules and saved/rebuilt
T normalization. Support the actual native time length. Keep the classifier's
normalization history separate from newly fitted covariance calibration.

Implement prescribed80/20 T split and static22/16/6 masks with partition-disjoint
nonfull subsets. Expose explicit train/validation/E access; no implicit E loading
in calibration or plan. Keep masks paired across all methods and persist IDs.

Tests must catch hidden raw channel contamination through preprocessing, labels
shifted by exclusions, channel permutation, sample-offset errors, deterministic
splits and illegal masks. A supervisor T-only native-adapter comparison on a
fixed trial sample is permitted after synthetic checks; it is a data comparison,
not a classifier score. Native recording downloads remain supervisor-owned.

Record computational differences from historical 0–4s/window-local inputs.
If the audited pipeline is not compatible with the planned missing-electrode
semantics, report the specific contradiction; do not silently change preprocessing.
Handoff includes commands, test counts, native-array evidence and limits.
