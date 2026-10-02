# Session 10 — Review the finished baseline package as a new collaborator

Read previous handoffs, inspect all new baseline code and tests, and independently check the scientific boundaries. This is a review/fix session, not new model work.

Run all baseline tests and an end-to-end synthetic smoke in a temporary directory. Check real/synthetic separation, train-only normalization, raw masking before all mixing, checkpoint reload, full/robust validation selection without test access, identity protection, output coverage and balanced aggregation. Fix concrete issues narrowly and add regression tests only where needed.

Verify legacy run.py --plan still prints 378 fits. Check missing prerequisites produce actionable failures without partial 'complete' records. Ensure a real task cannot silently fall back from CUDA to CPU or from GDF data to synthetic input.

Write docs/baselines/runbook.md with actual tested commands: environment preparation, resolved data paths, preflight, smoke, one A01/seed-0 pilot, reporting, all-participant runs, and cross-session config. No dependency installation or GPU scheduling is executed in this coding session. Describe how to resume and where selected checkpoints reside. State that benchmark scores are unknown until experiments run.

Write docs/baselines/review.md with findings, fixes, exact checks and remaining empirical questions. Link the new runbook in docs/README.md and onboarding. Do not rewrite historical investigation evidence as if numerical execution existed at its earlier review date.

Acceptance: independent integration tests exercise CLI plan/smoke/summarize and selected-checkpoint reload; the full test suite has no missing-dependency skips. Review must explain that this baseline architecture alone is not a paper-protocol reproduction.

Handoff: clear 'software ready / real pilot not run' distinction and the concrete evidence required before sessions 11–12.
