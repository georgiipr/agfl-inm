# Session 04 Analyse the completed cohort and issue verdicts

Required artifacts and bounded paths:

- `docs/candidate-followup/findings.md`
- `docs/candidate-followup/decision.md`

Review complete real27-task/five-arm evidence and all predeclared contrasts.
Independently recompute or cross-check participant means, paired deltas and the
screening rule from validated task records. Read selected histories and probes;
separate optimization-time metrics from clean train scores. Explain incomplete
or inconsistent evidence rather than inventing values or silently dropping rows.

Write a concise verdict for local_power, spatial_filterbank and the Transformer:
validation BA/log-loss, paired bootstrap uncertainty, positive participants,
parameters, train–validation gaps and secondary missing-electrode performance.
Keep original equal-participant aggregation and reused-validation caveats. Assess
probe gains against shuffled controls without treating one shuffle as a p-value.
No test scores, tuning, retraining or new code changes. At most one bounded
follow-up recommendation; no candidate passing is an acceptable conclusion.
