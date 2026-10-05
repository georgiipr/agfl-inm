# Encoder candidate reporting review

Session 09 implements independent validation of the real evidence layout and
validation-only aggregation. The `--summarize-only` command writes
`report/evidence.json`, `summary.md`, `validation_contrasts.csv`,
`per_subject_scores.csv`, `training_gaps.csv`, `probes.csv`, `robustness.csv`,
and `issues.csv`. A complete evidence manifest is emitted only for all 27
expected non-synthetic tasks, with all five arms and matching task/fit identities.
Missing, failed, malformed, duplicated, synthetic, or mismatched records remain
issues; they cannot contribute to a complete cohort mean. Individual verified
task results remain visible in partial tables.

The summarizer checks required schema/cohort identity, raw config digest,
source/package maps, original metadata and recording digest maps, seven
executed Boolean checks, ordered arm coverage, fit/task split and data IDs,
prediction and probe sample IDs, all 21 validation conditions, shared mask
hashes, selected epoch/history agreement, finite clean/probe/condition metrics,
and every checkpoint/history/NPZ digest listed in each result. Numeric NPZ files
are opened with `allow_pickle=False`. It does not load test data or test scores.
The checksum chain detects accidental corruption and incompatible reuse; it is
not a cryptographic signature against coordinated artifact rewriting.

Aggregation first averages each degraded condition's repeats in each fit,
then averages seeds within each participant, then weights the nine participant
means equally. Primary and secondary contrasts use the same participant/seed
rows. Bootstrap resampling draws participant indices and keeps each participant's
paired candidate/control values together. Intervals are exploratory; epoch
selection reuses validation and there are three predeclared contrasts without a
multiplicity correction. The screening flag applies the fixed 2 percentage point
BA mean and 6/9 positive participant rule, and only suggests independent
confirmation.

The changed shared study identity is a required interface repair: fit records,
task records, and `evidence.json` now share the global configuration/source/
package `study_id` required by the session-10 gate. Task-specific `split_id`,
`data_id`, recording hashes, original metadata hashes, and exact sample IDs
remain on each task and fit for input provenance and pairing. No old study
checkpoint, history, config, or manifest is changed. Use a fresh candidate
output if source, config, or installed package versions change.

The local linear probes describe ordered frozen feature information in a
particular fixed logistic model. A weak or nonconverged probe does not show that
the neural representation lacks useful nonlinear information. The shuffled-label
control is one diagnostic fit, not a p-value. Candidate comparisons also change
capacity, pooling, regularization, and optimization; the spatial Transformer
comparison changes the complete head. Validation results are not test-set or
external confirmation results.

The synthetic smoke now calls the same reporting path and demonstrates that a
five-arm synthetic task remains partial and cannot satisfy the real evidence
gate. Synthetic training verifies implementation paths only; it supplies no
classification measurement for the study.
