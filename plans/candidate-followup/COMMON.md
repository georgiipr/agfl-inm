# Instructions for follow-up sessions

Read AGENTS.md, this contract, the previous candidate contract, current task and
verified handoffs. Inspect Git status and preserve existing work. Implement only
the bounded session. No commits, pushes, installations, downloads, nested runners
or subagents. Use AGFL_PYTHON. No real training inside coding/review sessions.

Sessions01–02 may repair only the new candidate package/tests and create named
new config/docs. Existing plans, scripts, automation tests, old configs/results
and baseline/legacy packages are not implementation targets. Necessary adjacent
candidate-package repairs need explanation and regression checks. Sessions03–04
are evidence reviews: no scientific source/config changes after readiness freeze.

Use tiny synthetic CPU fits for software checks, never invent metrics or skip
mandatory tests. Separate optional CUDA capability checks and record unavailable
hardware honestly. Session02 may load the one real pilot input without fitting.
Missing real evidence stops review sessions before a model call.

Return exact structured JSON. completed requires a meaningful summary, actual
checks and blockers:[]. Unfinished implementation or failed mandatory checks
means blocked. Missing real input readiness remains explicit in readiness.json
and next_session_notes; do not conflate it with otherwise completed software.
A scientifically valid pilot review may say stop while the review session is
completed; the experiment launcher then refuses cohort execution.

Do not modify a plan/checker to pass its acceptance. Verify the intended behavior
through the public CLI as well as direct functions. Preserve all negative
findings, hashes and failed artifacts for review.
