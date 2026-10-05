# Session 05 Implement compact spatial Transformer

Bounded paths:

- `inm/encoder_candidates/transformer.py`
- `inm/encoder_candidates/models.py`
- `tests/encoder_candidates/test_transformer.py`
- `docs/encoder-candidates/architectures.md`

Implement only the contracted one-layer temporal Transformer over the existing
EEGNet front-end features. Inspect forward_features before adapting it. Preserve
the original model and provide a reloadable constructor/factory. All five factory
entries must now work. Shared front-end tensors must initialize identically to
the spatial reference for each seed, unaffected by the extra head parameters.

Use fixed sinusoidal temporal positions, 32 dimensions, four heads, FF width 64,
GELU/dropout 0.1, final norm, mean pooling, flags and linear classifier. Record
parameter counts and explain this is a new head comparison, not a published
Conformer reproduction or a pure test of temporal order.

Acceptance: positional-code correctness, feature/logit shapes, finite gradients
through front-end and attention, reproducible shared initialization, eval-mode
reload equality, hidden-NaN invariance, wrong/all-missing masks fail. Confirm
all five constructors round-trip. Do not claim it improves accuracy.
