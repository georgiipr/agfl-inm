# Session 01: protocol

Read ../CONTRACT.md and verified predecessor handoffs/receipts.

Deliverable: Freeze strict config, dependency-light plan/preflight and provenance; write protocol docs and tests.

Write scope ONLY: inm/task_driven_completion/__init__.py; inm/task_driven_completion/protocol.py; inm/task_driven_completion/__main__.py (plan/preflight only); configs/task-driven-completion.json; tests/task_driven_completion/test_protocol.py; docs/task-driven-completion/protocol.md; docs/task-driven-completion/session-01-handoff.md. No plans/acceptance/runners/historical writes.

Acceptance command (supervisor independently reruns):
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 01`

Include meaningful tests; zero discovered tests or skips do not pass. Session02 tests explicit observed design ridge reference/finite-difference gradients, hidden NaNs, preserved observations, paired initialization, frozen backbone state and nonzero input/factor gradient. Session03 tests repeated fits under changed ambient RNG/arm order, mask partition separation, epoch0 selection, numeric state reload, corruption/source change/incomplete reports. Session04 tests public CLI fresh/resume/change identity, all five strategies and prior completion/supervised-Tucker regressions.

Finish docs/task-driven-completion/session-01-handoff.md with exact commands/results, files and unresolved gates. No real fitting. Default deadline 1200 seconds from supervisor start.
