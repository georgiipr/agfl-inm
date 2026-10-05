# Session 05 Measure accessible class information with fixed probes

Implement `inm/encoder_audit/probes.py`,
`tests/encoder_audit/test_probes.py`, and `docs/encoder-audit/probes.md`.
Use verified legacy features/factors, not the spatial EEGNet activation cache.

1. Implement the three feature views and one shuffled-training-label control
   exactly as CONTRACT.md declares. Fit each scaler/classifier on training only,
   with fixed C=1 and no sweep, validation refit, or best-seed selection.
2. Infer cores with frozen saved factors using full-input masks. Preserve
   encoder/factor bytes. Save numeric probe state, feature layout, identities,
   convergence status, train/validation metrics, and per-sample probabilities/IDs.
3. Provide a loader/predictor that returns equivalent probabilities after
   saving. The ordered-feature probe must accept normalized feature tensors so
   session 06 can evaluate completion without another fit.
4. Document conclusions this probe can and cannot support. Window averaging
   changes dimension; low linear accuracy does not prove all label information
   is absent. A single real label shuffle is a sanity check, not a p-value.

Acceptance: a separable synthetic signal is learned; a designed label-independent
validation fixture does not leak labels; scaler statistics are training-only;
the fixed shuffle affects training labels only; invalid labels/nonconvergence
fail explicitly; save/reload equality and frozen-factor invariance hold. Run
artifact and probe tests. No real probe fitting in this session.
