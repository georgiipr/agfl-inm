# Session 04: completion and covariance-only training

Read contract, acceptance and verified handoffs. Default20minutes; checks<=60s.
Synthetic data only; no real supervised fitting by the worker.

Write scope: `published_covariance/completion.py`,
`published_covariance/training.py`, corresponding tests,
`docs/published-covariance/training.md`, `session-04-handoff.md`.

Implement native-length zero/fixed/learned completion from one training second
moment, matching the contract's SPD parameterization, ridge and 253 float64
parameters. Existing equations can be reused; old shape assumptions cannot.
Implement loss, optimizer, masks, deterministic batching, early stopping,
epoch0 eligibility and exact selected-state restoration. No E data access.

Only covariance updates. Keep the backbone frozen/eval while maintaining input
gradients; persist backend/RNG choices. CPU synthetic training must not mutate
global random/backend state unexpectedly. Covariance stays float64 even when
classifier inputs/parameters use float32.

Independent acceptance covers analytical solves, gradients, observed preservation,
hiddenNaN/Inf invariance, differentiable all-observed loss, fixed/learned epoch0
equality, gradient flow through frozen classifier, validation isolation and
frozen-state hashes through an actual synthetic optimizer step.

Handoff reports meaningful test outcomes and any precision/gradient failures.
Do not infer scientific effectiveness from a synthetic loss decrease.
