# Session 06 Train and reload selected classifiers

Bounded paths:

- `inm/encoder_candidates/training.py`
- `tests/encoder_candidates/test_training.py`

Implement fit_classifier using the fixed optimizer/LR schedule/CE/selection.
Use its own small evaluation helper; diagnostics from session 07 run after fitting.
Keep data-order RNG separate from initialization and dropout so every arm sees
identical ordered training batches. Respect full-input-only training. Implement
real/synthetic budget separation, minimum epoch, patience, and earlier-epoch tie.

Persist full weights-only-safe checkpoint, history, selected epoch and clean
train/validation metrics. Restore selected weights before final scoring. Check
finite losses/gradients and call model clipping. Record optimization-time metrics
separately. No test arrays exist in its input type. Hash/atomic-write artifacts;
never overwrite conflicting files. Reuse only fully verified compatible outputs.

Acceptance: tiny deterministic CPU learnability fixture (loss reduction, not a
promised real score), selection independent of last epoch/test sentinels, BA and
log-loss tie handling, min-epoch/patience boundaries, exact LR endpoints, paired
minibatches despite different parameter counts, eval BatchNorm/dropout, all-five
model reload probability equality and failure on corrupt state/NaNs. Tests use
few synthetic updates, never the real 250-epoch budget.
