# Session 03: execution

Read ../CONTRACT.md and verified predecessor handoffs/receipts.

Deliverable: Implement deterministic supervised completion training, historical replay integration, artifacts/resume, hierarchical reporting and public CLI smoke/run/summary.

Write scope ONLY: inm/task_driven_completion/training.py; inm/task_driven_completion/study.py; inm/task_driven_completion/reporting.py; inm/task_driven_completion/__main__.py; tests/task_driven_completion/test_training.py; tests/task_driven_completion/test_study.py; docs/task-driven-completion/session-03-handoff.md. No plans/acceptance/runners/historical writes.

Acceptance command (supervisor independently reruns):
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 03`

Include meaningful tests; zero discovered tests or skips do not pass. Session02 tests explicit observed design ridge reference/finite-difference gradients, hidden NaNs, preserved observations, paired initialization, frozen backbone state and nonzero input/factor gradient. Session03 tests repeated fits under changed ambient RNG/arm order, mask partition separation, epoch0 selection, numeric state reload, corruption/source change/incomplete reports. Session04 tests public CLI fresh/resume/change identity, all five strategies and prior completion/supervised-Tucker regressions.

Finish docs/task-driven-completion/session-03-handoff.md with exact commands/results, files and unresolved gates. No real fitting. Default deadline 1200 seconds from supervisor start.
