# Session 03: replay

Read ../CONTRACT.md and all earlier handoffs in docs/completion-transformer/.

Implement replay.py: verify historical task/fit/checkpoint/history/prediction checksums and metadata, restore only the two fixed models safely, verify relevant saved source/package identities, compare original full-input validation predictions, and use existing prepare_subject with outputs only in NEW task staging. Expose replay/data interface for session04. Synthetic fixture tests must reject corrupt artifacts, incorrect split/data/normalization/IDs, source drift and mismatched replay predictions. Read-only real artifact inventory allowed, no real factor fit. Fail honestly if old inputs cannot replay.

Use .venv/bin/python; set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for CPU checks.
Keep scientific imports lazy in protocol/CLI planning. Run meaningful synthetic
tests using unittest discovery under tests/completion_transformer. Do not alter
old packages or workflows. No real training or factor fitting.

Write docs/completion-transformer/session-03-handoff.md listing files, APIs,
commands and outcomes, limitations, and unresolved blockers. Never mark a test
passed without executing it. Return the same concise account to the supervisor.
