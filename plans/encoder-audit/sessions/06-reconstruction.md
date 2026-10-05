# Session 06 Compare reconstruction with classifier behavior

Implement `inm/encoder_audit/reconstruction.py`,
`tests/encoder_audit/test_reconstruction.py`, and
`docs/encoder-audit/reconstruction.md`. Read `_reconstruction_scores` in the
legacy study to preserve the exact hidden-entry NRMSE definition.

1. Generate the declared paired validation banks (full, static/dynamic random
   16/6; five repeats). Using one fitted ordered-feature probe, compare full
   features, zero-fill, and Tucker completion. Keep its training scaler fixed.
2. Compute condition-level hidden squared-error/energy sums before taking the
   NRMSE square root. Also record per-trial error/log-loss and class-stratified
   summaries. Keep full-input NRMSE null. Handle zero energy and missing classes
   without fabrication. Do not average per-trial ratios as the primary NRMSE.
3. Report paired BA/log-loss/recall changes and mask identities. No new probe
   fitting, reconstruction-based rank selection, or use of 11/spatial/test scores.
   Labels are accessed only after completion to evaluate classifications/groups.

Acceptance: explicit hand-computed NRMSE fixture including unequal trial energy;
zero-fill score 1 for nonzero hidden energy; full-input null; observed entries
preserved; hidden NaN invariance; identical masks across methods; labels cannot
affect reconstruction; forbidden validation conditions rejected. Run probe and
reconstruction tests. Handoff metric keys and repeat-aggregation semantics.
