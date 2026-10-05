# Session 04 Re-evaluate selected EEGNet checkpoints cleanly

Implement `inm/encoder_audit/checkpoints.py`,
`tests/encoder_audit/test_checkpoints.py`, and `docs/encoder-audit/checkpoints.md`.
Read `inm/baselines/training.py` and the corrected baseline review. Do not alter
or refit the original classifiers.

1. Restore all three saved neural arms with their constructor, normalization,
   class/channel order, and selected epoch. Use aligned inputs from session 03.
2. Evaluate clean full-input training and validation under `eval()` and no-grad.
   Report BA, accuracy, macro-F1, log-loss, recalls/confusions, and train–validation
   gaps. Replay selected-epoch history within the declared numerical tolerances;
   preserve discrepancies, not averaged excuses. Verify input checkpoints unchanged.
3. Keep optimization-time masked/dropout metrics separately labeled. Explain
   that the legacy encoder selection score is historical metadata; its missing
   pretraining head prevents a new clean classifier replay. Probes are the next
   stage, not a substitute labeled as that missing model.

Acceptance: evaluation disables dropout and freezes BatchNorm; parameter and
buffer state is unchanged; reload predictions agree; wrong epoch/class order
fails; recorded versus recomputed metrics are checked; test data cannot affect
outputs. Run alignment and checkpoint tests. Handoff stable metric schema and
the policy for unsupported historical checkpoint formats.
