# Real pilot review

A01/seed0 completed successfully on CPU. All 126 saved evaluation records,
completion state/history, deterministic paired masks and original checkpoint
provenance passed verification. Balanced accuracy and log loss were independently
recomputed from numeric predictions. All 1,411 protected files remain unchanged.
Tucker fitted 30 epochs on training raw only; no classifier was retrained.

The operational decision is **proceed** under the existing user authorization.
Full-input predictions remain unchanged. Mean degraded Transformer validation BA
is 34.90% zero-fill, 40.34% Tucker and 63.44% covariance. These are descriptive
one-task results; no cohort inference or setting selection is made. Continue
the predeclared cohort unchanged, including the covariance control. Exact task
checksum and scores are in pilot-review.json.
