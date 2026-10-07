# Session 04 handoff: covariance completion and training

**Decision:** the native-length TensorFlow completion and covariance-only
training implementation passes the independent synthetic completion and
training acceptance suites. No real covariance fit, classifier outcome scoring,
or E-data access occurred. These checks do not support a scientific effectiveness
claim.

`published_covariance/completion.py` provides zero fill, hidden-entry MSE and
conditional Gaussian completion. It uses a 253-entry float64 Cholesky
parameterization, `S + 1e-6 I` initialization, softplus diagonal, the specified
mean-diagonal ridge, safe hidden-entry selection, exact observed-value
preservation and a differentiable masked block solve. The block solve is
algebraically equivalent to solving each observed principal block and never
forms a matrix inverse. Fixed and learned instances initialize identically.

`published_covariance/training.py` fits the uncentered second moment from
training inputs only, checks participant T-trial IDs and disjoint validation
IDs, creates deterministic local epoch order, and applies equal-probability
22/16/6 masks. Validation scores the same five 16-channel and five 6-channel
conditions at every epoch. Training uses the frozen classifier in inference
mode, passes gradients through classifier inputs, updates only covariance with
the declared Adam/objective/clipping values, logs meaningful per-epoch metrics,
and restores the exact selected state. The real 100/10/20 epoch budget is fixed;
only synthetic calls may shorten it.

Checks run with TensorFlow 2.15.1 and NumPy 1.26.4 using
`.venv-published-covariance/bin/python` on CPU:

- `-m unittest tests.published_covariance.test_completion tests.published_covariance.test_training -v` — 6 tests passed. This includes native-length analytical solve comparisons, observed bit preservation, hidden NaN sanitization, a nonzero-target hidden-MSE reference with large observed values, nonzero covariance gradients, checkpoint tie ordering, an actual optimizer update, exact epoch-zero restoration, immutable validation/frozen weights, and Python/NumPy RNG isolation.
- `PYTHONPATH=. .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_completion.py` — 4 independent acceptance tests passed, including five finite-difference gradient coordinates.
- `PYTHONPATH=. .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_training.py` — 1 independent acceptance test passed, including actual update detection and exact epoch-zero restoration.

The complete local `tests/published_covariance` suite passed 17 tests after the
hidden-MSE repair.

The synthetic training fixture uses 8 trials, two epochs and short time arrays.
The acceptance training fixture also passed on 1,125-sample arrays. There is no
GPU available in this worker environment. The full local data/classifier suites,
real participant fitting, final artifact generation and E evaluation were not
run. The classifier and all prior research artifacts remain unchanged.
