# Task-driven completion sessions

Follow [CONTRACT.md](CONTRACT.md). Five strategies before a frozen EEGNet–Transformer;
27 participant/seed tasks and 54 supervised completion fits in the future real study.
This authorization is implementation and synthetic verification only.

1. [Protocol](sessions/01-protocol.md)
2. [Models](sessions/02-models.md)
3. [Execution](sessions/03-execution.md)
4. [Integration](sessions/04-integration.md)

One fresh worker at a time, 20-minute deadline, supervisor checks at <=60-second
intervals, independent acceptance and protection checks before advancement.
State/receipts: `.session-runs/task-driven-completion/`.
Handoffs/runbook: `docs/task-driven-completion/`.
Worker uses current model unless user chooses otherwise. A built-in watchdog
exists only while the supervisor is active. Real pilot/cohort are separate.
