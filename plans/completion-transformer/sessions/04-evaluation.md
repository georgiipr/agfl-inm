# Session 04: evaluation

Read ../CONTRACT.md and all earlier handoffs in docs/completion-transformer/.

Implement study.py, reporting.py and __main__.py: lazy --plan/--preflight, synthetic --smoke in explicit fresh output, explicit --task-index for real replay + completion factor fitting, --summarize-only. Evaluate six paired cells, save numeric probabilities/labels/sample IDs/masks and completion state plus hashes. New identity refuses incompatible/partial overwrite, verifies complete resume, excludes synthetic from real reports. Compute predeclared hierarchical primary and secondary contrasts, participant bootstrap, incomplete coverage and log loss. Reuse prior APIs; no neural fitting. Test pairing, aggregation, artifacts, smoke and CLI/resume. No real task execution.

Use .venv/bin/python; set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for CPU checks.
Keep scientific imports lazy in protocol/CLI planning. Run meaningful synthetic
tests using unittest discovery under tests/completion_transformer. Do not alter
old packages or workflows. No real training or factor fitting.

Write docs/completion-transformer/session-04-handoff.md listing files, APIs,
commands and outcomes, limitations, and unresolved blockers. Never mark a test
passed without executing it. Return the same concise account to the supervisor.
