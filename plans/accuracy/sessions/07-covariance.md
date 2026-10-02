# Session 07 — Classical covariance baseline using existing dependencies

Implement CovarianceClassifier and wire it into study execution. Use NumPy/SciPy/scikit-learn already in requirements; do not add pyriemann for this first baseline.

For each full-input filtered trial, transpose to [time,channel], estimate Ledoit–Wolf shrinkage covariance, floor eigenvalues relative to trace, compute its symmetric matrix logarithm, and vectorize the upper triangle, multiplying off-diagonal entries by sqrt(2). This is a LOG-EUCLIDEAN covariance baseline; do not call it an affine-invariant Riemannian-mean/tangent implementation.

Train StandardScaler + multinomial-capable LogisticRegression (L2, fixed C=1 initially) using only training trials. Pin/check actual sklearn API available; no hidden hyperparameter search or default-version assumptions. Use unnormalized filtered signals from PreparedData; do not fit scaling on held-out data. Save reloadable numeric fitted state and class mapping, or clearly document a trusted-local serialization format.

The classical arm is full-input only. Emit one full row per evaluation partition and explicit full-only coverage; it must not be marked incomplete for absent degraded rows. Neural coverage is unchanged.

Acceptance: finite features on near-singular covariance, symmetric matrix log and vector weighting, probability columns follow class order, scaler stats unchanged by validation/test inputs, model roundtrip reproduces probabilities, tiny four-class pipeline runs, classical versus neural coverage correctly represented.

Scope: covariance implementation, narrow study dispatch, tests. No unsupported claim that the baseline is itself SOTA.

Handoff: preprocessing distinction, checkpoint format and report coverage implications.
