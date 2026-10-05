# Session 03 Audit trial alignment and feature replay

Implement `inm/encoder_audit/alignment.py` and
`tests/encoder_audit/test_alignment.py`. Consume session 02's verified views.
Inspect `inm/data.py`, `agfl/datasets/splits.py`, the baseline data preparation,
and the legacy `_calibrate` function. Do not call code that writes original splits.

1. Rebuild source trial IDs and exclusions using original settings; verify cue
   alignment, class/channel order, sample counts, and persisted split membership.
   Distinguish a matched cross-study cohort from two valid but unmatched inputs.
2. Recompute raw normalization from training indices only and compare to saved
   values. Apply each study's own saved convention. Reconstruct the legacy frozen
   encoder and replay training/validation cached features, including the saved
   training feature mean/std. Never use validation statistics to repair a mismatch.
3. Report maximum numerical errors, tolerances, BatchNorm/dropout state, and
   invariant violations. A replay failure blocks dependent analyses. Record
   cross-study mismatch separately; do not drop difficult trials to obtain a match.

Acceptance: synthetic cue/label shifts, reordered IDs, changed normalization,
and validation-contaminated statistics fail. Cached/replayed features agree;
different batch sizes give equivalent frozen outputs; perturbing another
electrode/window does not affect the local observed feature; hidden NaNs do not
leak. Use mocked loader events, not a downloaded GDF. Run artifact and alignment
tests. Handoff which mismatches block within-study versus cross-study comparisons.
