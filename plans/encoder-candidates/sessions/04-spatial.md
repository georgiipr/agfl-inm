# Session 04 Implement spatial reference and filter bank

Bounded paths:

- `inm/encoder_candidates/spatial.py`
- `inm/encoder_candidates/models.py`
- `tests/encoder_candidates/test_spatial.py`
- `docs/encoder-candidates/architectures.md`

Wrap the current spatial EEGNet reference and implement the fixed six-band
FIR/spatial/log-variance model. Complete the corresponding factory entries and
constructor reload. Describe shape changes and distinguish latent spatial
components from original electrodes. Use fixed coefficient buffers and no
per-example normalization before log-variance. Apply availability flags at the
head as contracted. Do not import external implementations or expand bands.

Acceptance: analytical/reference FIR and log-variance checks, passband versus
stopband response on synthetic sinusoids, filter buffers have no gradients,
spatial filters/classifier do receive gradients, max-norm clipping works, and
window-local FIR does not cross masked boundaries. Check hidden-NaN invariance,
observed nonfinite rejection, None/full equivalence, valid logits and reload.
Do not impose channel locality on learned spatial activations.
