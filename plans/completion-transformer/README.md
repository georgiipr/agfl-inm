# Raw completion with the EEGNet Transformer

Test the combination without changing either historical experiment. Read
[CONTRACT.md](CONTRACT.md) for the frozen six-cell replay design and evidence limits.

| Session | Deliverable |
|---|---|
| [01](sessions/01-protocol.md) | Branch review, protocol, configuration and provenance |
| [02](sessions/02-completion.md) | Frozen raw completion and mask-preserving adapters |
| [03](sessions/03-replay.md) | Verified historical checkpoints and train/validation input |
| [04](sessions/04-evaluation.md) | Paired evaluation, aggregation and CLI |
| [05](sessions/05-integration.md) | Integration checks, readiness and real-run runbook |

Execute one fresh gpt-6-luna/high session at a time. The supervisor records a
1200-second deadline, waits for its output, independently runs acceptance checks,
reviews the patch and handoff, and records proceed/stop before launching another.
A timeout interrupts the worker; errors preserve artifacts and stop advancement.
No automatic retries or automatic real experiments. Model availability is not
a price claim. This session uses the built-in agent tools; the reusable skill
also describes the existing CLI-runner alternative.

Receipts and review logs: .session-runs/completion-transformer/.
Persistent handoffs and findings: docs/completion-transformer/.
The working tree already contains earlier uncommitted studies; preserve them.

Real execution is a separate pilot / review / cohort sequence after implementation.
The default authorized work here is software and synthetic verification unless
the user additionally chooses real execution. No experimental result is promised.
