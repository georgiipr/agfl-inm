# Session 02 Verify CLI execution and freeze readiness

Required artifacts and bounded paths:

- `inm/encoder_candidates/protocol.py`
- `inm/encoder_candidates/__main__.py`
- `tests/encoder_candidates/test_execution.py`
- `configs/encoder-candidates-reproducible.json`
- `docs/candidate-followup/readiness.json`
- `docs/candidate-followup/execution.md`

Fix and verify safe existing-output config validation and public CLI sequencing.
Remove dependence on an unverified temporary config trick; keep path containment,
immutable inputs and strict task/fit identity checks. Allow adjacent candidate
study/reporting repairs only if a reproduced CLI failure requires them. Do not
change the reporting schema or scientific settings to bypass failures.

Create the new config by copying the old scientific settings and changing only
name/output. Exercise actual subprocesses: first task, second task, completed
resume, summary, changed source/config/package, corrupt/partial outputs. Use
explicit synthetic reduced cohort/budgets for these checks, never real fits.
Run the full candidate suite and baseline regressions. Run all-five CPU smoke,
then load real A01/seed0 once to verify its input checks without model training.
Record any numerical warnings that affect validity; do not hide nonconvergence.

Write readiness.json and hashed verification logs per CONTRACT.md after final
source freeze. CPU is mandatory; list cuda only after tiny synthetic repeated-fit
checks in a subprocess configured before CUDA initialization. If unavailable,
keep cpu readiness and disclose cuda as unverified. Missing real inputs allow
software completion with blocked readiness, never a fabricated ready flag.

Write execution.md with the exact follow-up and experiment commands. Explain
135 neural/108 probe budgets, pilot inspection, explicit cohort run, fail-stop
behavior, and that completing this session has not trained real EEG models.
