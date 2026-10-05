# Session 07 Add probes and paired robustness diagnostics

Bounded paths:

- `inm/encoder_candidates/diagnostics.py`
- `tests/encoder_candidates/test_diagnostics.py`

Implement evaluate_selected and the two fixed probes per selected local encoder.
Freeze local features before extraction; StandardScaler fits training only. Save
numeric NPZ models and per-sample probabilities and verify reload. Preserve a
shuffled-training-label control without requiring a lucky exact chance result.
All probe fits and metrics use train/validation only. Nonconvergence is explicit.

Score full validation plus the four contracted random-loss conditions, five
repeats each, from masked RAW data for all five models. Use paired deterministic
banks and original channel IDs. Record scenario/repeat/mask hashes; never use
spatial feature caches to represent raw loss. All-missing windows fail. Do not
add Tucker, 11-channel/spatial losses, or mixed training.

Acceptance: scaling/shuffle isolation, label poisoning, reload probability
agreement, expected fit counts, convergence failures, finite metrics, paired
mask hashes, and raw-vs-post-mixing mask counterexample. Verify frozen locality
for local features and changing observed input can affect spatial activations.
Record true train labels in scores even for shuffled fits.
