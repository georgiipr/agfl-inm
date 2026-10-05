# Session 01 Repair and prove training reproducibility

Required artifacts and bounded paths:

- `inm/encoder_candidates/training.py`
- `tests/encoder_candidates/test_reproducibility.py`
- `docs/candidate-followup/reproducibility.md`

Reproduce the uncontrolled CPU dropout RNG first with a failing regression.
Same seed0 and initial spatial_eegnet weights previously yielded different
optimization histories under ambient torch seeds123 and456. Implement scoped
training RNG and deterministic settings, stable explicit seeds, and provenance.
Inspect all model constructors for CUDA RNG side effects. Necessary RNG-only
repairs in local.py/spatial.py/transformer.py/models.py or a new randomness.py
are permitted; architecture tensors and hyperparameters must remain unchanged.

Follow CONTRACT.md repeatability, arm-order, subprocess and exception tests.
Compare actual short training runs, not initialization alone. Include all-five
models on CPU and preserve paired minibatch streams. Test new RNG metadata
survives checkpoint restore and rejects incompatible reuse. CUDA is separate,
optional hardware verification; do not add skipped CPU tests.

Acceptance: the original failure now passes, all listed tests pass, existing
candidate tests remain compatible. Document exact seeds/settings, limitations,
and any CUDA capability still unverified. No real fits; do not clear old logs.
