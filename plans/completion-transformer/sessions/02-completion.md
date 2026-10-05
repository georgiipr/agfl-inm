# Session 02: completion

Read ../CONTRACT.md and all earlier handoffs in docs/completion-transformer/.

Implement completion.py and adapters.py. Train-only frozen Tucker and channel-covariance controls, exact observation preservation, local windows, scoped RNG, and adapters for both existing backbones that preserve original availability flags while actually using filled values. Tests include explicit ridge oracle, hidden NaN/Inf, all-missing rejection, original zero/full logits equality, completion influence, factor state reload and unchanged classifier weights. Coordinate public APIs in handoff.

Use .venv/bin/python; set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for CPU checks.
Keep scientific imports lazy in protocol/CLI planning. Run meaningful synthetic
tests using unittest discovery under tests/completion_transformer. Do not alter
old packages or workflows. No real training or factor fitting.

Write docs/completion-transformer/session-02-handoff.md listing files, APIs,
commands and outcomes, limitations, and unresolved blockers. Never mark a test
passed without executing it. Return the same concise account to the supervisor.
