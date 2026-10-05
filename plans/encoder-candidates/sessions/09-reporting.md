# Session 09 Report comparisons and document real execution

Bounded paths:

- `inm/encoder_candidates/reporting.py`
- `inm/encoder_candidates/study.py`
- `inm/encoder_candidates/__main__.py`
- `tests/encoder_candidates/test_reporting.py`
- `docs/encoder-candidates/runbook.md`
- `docs/encoder-candidates/review.md`

Implement strict report validation/aggregation and wire --summarize-only and
smoke reporting. Match the session 10 evidence gate contract exactly. Validate
five-arm coverage, paired splits/IDs/masks, every checkpoint/history/NPZ checksum,
selected epoch and finite metric fields; exclude failed or synthetic tasks from
real evidence and withhold incomplete cohort means. Preserve negative results.

Report the three predefined contrasts, all per-subject scores, parameters,
training gaps, ordered/shuffled probes and secondary robustness. Average repeats,
then seeds, then participants equally; bootstrap participants with paired arms.
Apply the predefined promising-candidate rule without promoting exploratory
intervals to confirmation. Do not load or summarize test scores.

Write precise commands for plan/preflight, fresh temporary CPU smoke, then ONE
real A01/seed0 task (five neural and four CPU fits), inspection, remaining 26 tasks
sequentially, summary and session 10. Human execution stays outside coding
sessions. Explain source freeze, new output after source/config/package changes,
and why old checkpoint mismatches do not authorize modifying old artifacts.

Acceptance: unequal participant trial counts, missing/failed/duplicate arms,
mask/split mismatches, tampered artifacts, partial reports, synthetic exclusion,
expected CIs on deterministic fixtures, and positive/negative screening examples.
Run all candidate and baseline tests; perform tiny all-five-model smoke and check
it cannot pass the real evidence gate. No real recordings or GPU jobs.
