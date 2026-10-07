# Session 03: frozen published classifier

Read contract, acceptance and verified predecessor handoffs. Default20minutes;
one fresh worker, checks<=60seconds. No covariance fitting or E scoring.

Write scope: `published_covariance/classifier.py`, narrowly required attributed
model definitions under `published_covariance/`, classifier tests,
`docs/published-covariance/classifier.md`, `session-03-handoff.md`.

Load all nine audited checkpoints using exact model definitions and input shapes.
Retain native output semantics and mapping. Freeze parameters and normalization
buffers; dropout and BatchNorm always evaluate. Enable input gradients for later
covariance optimization. Do not append availability flags or alter spatial layers.

Prefer native inference. Any conversion must map every tensor, including BN
epsilon/statistics, depthwise/separable kernels, padding, activation and pooling.
Do not accept a port based on matching parameter counts or top-1 scores alone.

Provide forward and input-gradient parity tests against the native implementation,
with fixed synthetic inputs. Supervisor also compares native/wrapped predictions
on a fixed T-only sample for each subject, using the sealed tolerances. This is
technical parity, not an accuracy-based admission or model-selection gate.

Exact full-input bypass and frozen-state invariance are mandatory. Record framework,
device and precision behavior, retained licenses and any limitations. Failure
preserves diagnostics and stops advancement; no unverified substitute checkpoint.
