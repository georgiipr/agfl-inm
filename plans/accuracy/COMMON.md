# Instructions for every implementation session

You are executing one bounded task from the user's approved accuracy improvement
plan. Implement the current session only. Do not run future sessions, spawn
subagents, broaden the architecture, or launch the real study. You may inspect
earlier files and make small prerequisite repairs when required; explain them.

1. Read `AGENTS.md` if present, `plans/accuracy/CONTRACT.md`, the current session
   instructions, and supplied prior handoffs. Inspect `git status` before edits.
2. Preserve all existing user work, including untracked documentation. Do not
   commit, push, reset, delete experiments, or modify plan/runner/checker files.
3. The original v2 experiment retains its existing scientific semantics. The
   user has authorized a separate simpler baseline study; the explicit new-study
   exceptions in CONTRACT.md supersede legacy-only invariants for that study.
4. Use the supplied `AGFL_PYTHON` interpreter. Do not install packages or download
   data automatically. If numeric dependencies or other required prerequisites
   are absent, return `blocked` with an exact actionable reason. Do not skip tests
   or label unexecuted checks as passed. Never fabricate accuracy measurements.
5. Add only the tests required by the session. Use standard-library `unittest`
   rather than introducing a new test dependency. Keep fixtures synthetic and
   tiny; no network or real GDF recordings in automated checks. All tests should
   run on CPU. Scientific fitting in tests must finish quickly.
6. Run the current session's acceptance checks and the tests of any modified
   earlier module. Inspect the final diff. Changes must be reviewable and small.
7. The final response MUST be the JSON object requested by the output schema:
   exact session ID, status `completed` or `blocked`, a concise summary, changed
   file paths, actual check commands/results, blockers, and next-session notes.
   A completed response requires all task requirements and checks to pass.
   Include API names and material decisions in next-session notes, not a diary.

The runner performs its own tests and only advances on verified completion.
If the task cannot fit in this session, preserve useful partial work and return
`blocked` describing what remains; a stronger model can retry the same task.
