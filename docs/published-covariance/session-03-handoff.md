# Session 03 handoff: frozen published classifier

**Decision:** proceed to the covariance implementation session. The exact
native wrapper passed synthetic parity and gradient checks for all nine audited
run-1 checkpoints. No covariance fit, E scoring or outcome metric was run.

`published_covariance/classifier.py` implements `FrozenClassifier(subject,
assets_root=...)`. It rejects unknown subjects, non-identity source manifests,
changed source/license/checkpoint bytes, symlinked/escaping assets, invalid
input shape or dtype, and nonfinite input. It loads the four-class native graph
only after identity checks, exposes the Keras model at `.model`, freezes all
weights and BatchNorm buffers, forces inference behavior even if called with
`training=True`, and retains gradients to the input. The native softmax and
`[B,22,1125]` to `[B,4]` interface are preserved. The minimal graph definitions
are attributed in `published_covariance/atcnet_native.py` under Apache-2.0.

Checks run with `/home/kalexu97/Projects/AGFL/.venv-published-covariance/bin/python`
(TensorFlow 2.15.1, NumPy 1.26.4):

- `-m unittest tests.published_covariance.test_classifier -v` — exit 0; 2
  tests passed, covering A01 input validation, probability shape/normalization,
  nonzero finite input gradients and frozen-state behavior through a
  `training=True` call.
- `plans/published-checkpoint-covariance/acceptance_classifier.py ClassifierAcceptance.test_all_nine_synthetic_forward_gradient_frozen`
  — exit 0; the independent supervisor check passed for A01–A09 on its fixed
  synthetic input. It compared probabilities at abs `1e-5` / rel `1e-4`, exact
  predicted labels, gradients at abs `1e-5` / rel `1e-3`, exact repeated
  inference, and unchanged frozen weights.

The worker environment had no CUDA device; these checks ran on CPU. The separate
supervisor T-only input parity case was not run by this session. CUDA execution,
real fitting, E evaluation, and covariance behavior remain unverified here.
Checkpoint provenance remains grade B, including unresolved per-checkpoint
selection lineage and prior project E exposure.
