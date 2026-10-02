# Session 09 — Diagnose before changing Tucker mathematics

Extend the stdlib legacy diagnostics from session 01 to build a stage comparison from existing artifacts. Inspect encoder_history.json/calibration.json and per-arm history/result records. Compare full-input VALIDATION scores for the original shared pretraining model, retrained baseline head and Tucker head. Keep test metrics in separately labeled output, excluded from architecture recommendations.

Handle different selected epochs honestly: explain that comparisons of independently selected validation maxima are diagnostic and selection-biased, not independent test evidence. If common held-out raw predictions are not saved, do not pretend to reconstruct them from summary metrics. Report missing score sources explicitly.

Write docs/baselines/legacy-diagnosis.md explaining the decision tree: weak pretraining suggests representation/training/data issues; strong pretraining but weak new baseline suggests head/normalization/optimization; baseline strong but core weak suggests compression/regularization. Treat these as hypotheses. Include tensor rank/ridge and the 2816→64 entry reduction as context, not proof.

Acceptance: fixtures mixing val/test and selected epochs; valid matched calibration IDs; corrupted/mismatched IDs; absent calibration; no numeric packages needed for summary extraction. Never pair a new spatial EEGNet with a legacy Tucker head as if features matched.

Scope: diagnostics/report document and tests only. No real training, tensor optimization changes or automatic claims based on absent results.

Handoff: exact command/API for viewing stage diagnostics and available/unavailable evidence.
