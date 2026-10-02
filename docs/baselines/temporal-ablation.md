# Temporal head ablation

The real paired cohort is now complete. See
[temporal-results-review.md](temporal-results-review.md) for the validation
decision and the next diagnostic run. The specification below describes the
predeclared design and decision rule.

This controlled ablation asks whether a small convolution over EEGNet's learned
temporal feature sequence improves the masked CNN baseline. The flatten head
remains the default and within-study comparator. The ablation config writes to
`results/baselines-temporal-v1/`, separate from the baseline and legacy outputs.

The temporal head applies Conv1d to the `[time, F2]` feature sequence with
width 8, kernel 3, and dilation 1, followed by ELU and mean pooling over time.
The resulting representation feeds the same four-class linear readout. The
mask-conditioned classifier continues to append the original 88 channel/window
flags. With the declared 22-channel, four-window EEGNet settings, the masked
flatten model has 3,796 parameters and the temporal-head model has 2,236.
Parameter counts are recorded per result as well.

Both heads share the same temporal and spatial EEGNet encoder, raw `torch.where`
masking, full/mixed training regimes, data splits, preprocessing, optimizer and
epoch budget, validation selection policy, mask banks, and evaluation coverage.
The only planned model change is the readout head. The temporal head's settings
are part of each checkpoint constructor and the arm name. Reports keep every
head/regime combination under a distinct arm key and record `head_type`; paired
rows name both heads explicitly.

## Validation decision rule

Use only the configured `full` validation balanced accuracy to choose between
heads. For each matched regime (full or mixed), compute temporal minus flatten
validation balanced accuracy for the same subject, seed, data/split identity,
and mask hashes. Average seeds within participant, then weight participants
equally. Require a complete declared cohort for a cohort decision. A tie in
balanced accuracy is resolved by the corresponding validation log loss. Do
not use test rows or pick a head from one participant's result. Degraded
validation rows are diagnostic and do not alter this rule. No effect or benefit
is claimed before the paired ablation is run and reviewed.

## Paired run commands

Set `AGFL_PYTHON` to the project's scientific interpreter as described in the
[runbook](runbook.md). Run each task once; every task fits both head types for
both regimes on the same prepared split. These commands are provided for the
handoff and were not launched in this implementation session.

```bash
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines-temporal.json --plan
for task in $(seq 0 26); do
  "$AGFL_PYTHON" -m inm.baselines --config configs/baselines-temporal.json --task-index "$task" --device cuda || break
done
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines-temporal.json --summarize-only
```

The evidence needed for a decision is a complete real report with readable
histories and selected checkpoints for all 27 tasks and all four arms, exact
paired mask/data/split identities, selected epochs, and validation metrics for
both heads. Test metrics remain a separate final description and cannot affect
the head choice. The session-11 implementation does not run these experiments.
