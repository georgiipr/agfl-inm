# Session 01: protocol

Read ../CONTRACT.md and all earlier handoffs in docs/completion-transformer/.

Review both branch implementations and historical evidence. Write docs/completion-transformer/branch-review.md and protocol.md, fixed configs/completion-transformer.json, and dependency-light inm/completion_transformer/protocol.py plus __init__.py. Implement strict config validation, task plan, source/config/package identity, and read-only input inventory. Do not load real EEG or checkpoints numerically. Test config/path/identity behavior. Document historical 64.87% vs 59.50% BA as cited, not recomputed.

Use .venv/bin/python; set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for CPU checks.
Keep scientific imports lazy in protocol/CLI planning. Run meaningful synthetic
tests using unittest discovery under tests/completion_transformer. Do not alter
old packages or workflows. No real training or factor fitting.

Write docs/completion-transformer/session-01-handoff.md listing files, APIs,
commands and outcomes, limitations, and unresolved blockers. Never mark a test
passed without executing it. Return the same concise account to the supervisor.
