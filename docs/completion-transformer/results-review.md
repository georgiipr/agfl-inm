# Real completion experiment results — 2026-10-05

The 27-task experiment is complete: nine participants, three paired seeds, two frozen backbones and three input strategies. All 3,402 condition/repeat records passed verification. No neural classifier was retrained, no setting was changed, no task was retried, and no test partition was scored.

Tucker completion produces a small missing-input gain for the Transformer, but misses the predeclared primary screen. The simpler covariance completion control is substantially stronger. The backbone interaction does not establish an additional Tucker benefit for the Transformer.

## Equal-participant validation results

These means average the four degraded conditions equally: static and dynamic electrode loss with 16 or six retained electrodes. Mask repeats are averaged first, then three seeds within each participant, then all nine participants equally. Full-input performance is identical across completion strategies.

| Backbone | Strategy | Full-input BA | Mean degraded BA | Mean degraded log loss |
|---|---|---:|---:|---:|
| EEGNet | zero | 59.50% | 34.43% | 1.650 |
| EEGNet | tucker | 59.50% | 36.59% | 1.408 |
| EEGNet | covariance | 59.50% | 48.61% | 1.147 |
| EEGNet + Transformer | zero | 64.87% | 34.17% | 2.317 |
| EEGNet + Transformer | tucker | 64.87% | 36.01% | 2.178 |
| EEGNet + Transformer | covariance | 64.87% | 49.57% | 1.465 |

## Predeclared paired contrasts

Effects below are percentage points of balanced accuracy. Positive values favor the first named strategy. Intervals resample participants (2,000 replicates, seed 20261005).

| Contrast | Mean effect | Exploratory 95% interval | Positive participants |
|---|---:|---:|---:|
| Transformer: Tucker minus zero | +1.84 | [+0.68, +3.29] | 8/9 |
| EEGNet: Tucker minus zero | +2.16 | [+1.39, +3.02] | 9/9 |
| Transformer: Tucker minus covariance | -13.56 | [-18.04, -8.87] | 0/9 |
| Tucker benefit: Transformer minus EEGNet | -0.32 | [-1.14, +0.62] | 3/9 |

The primary rule required both a gain of at least 2 percentage points and at least six positive participants. The Transformer gain is 1.84 points with eight positive participants: the participant-count criterion passes, the mean-gain criterion fails. The threshold was not relaxed after seeing results.

The interaction is −0.32 points, with an interval crossing zero. Tucker helped the EEGNet reference by 2.16 points, versus 1.84 for the Transformer. This experiment therefore does not support a special advantage from combining Tucker completion with the Transformer readout.

Covariance completion gives the Transformer 49.57% degraded BA versus 36.01% with Tucker, a 13.56-point advantage; its participant mean exceeds Tucker for all nine participants. Its advantage over zero-fill is 15.40 points descriptively. EEGNet with covariance reaches 48.61% and has lower degraded log loss than the covariance Transformer (1.147 versus 1.465). The small accuracy difference between these two covariance backbones alone does not justify declaring the Transformer superior.

## Participant variation

| Participant | Transformer Tucker benefit | EEGNet Tucker benefit |
|---|---:|---:|
| A01 | +2.257 pp | +2.928 pp |
| A02 | +1.740 pp | +0.957 pp |
| A03 | +0.517 pp | +0.982 pp |
| A04 | -0.115 pp | +2.129 pp |
| A05 | +2.596 pp | +1.667 pp |
| A06 | +0.011 pp | +1.530 pp |
| A07 | +0.037 pp | +0.946 pp |
| A08 | +6.413 pp | +4.034 pp |
| A09 | +3.105 pp | +4.244 pp |

## Interpretation and next step

These are exploratory within-participant T-session validation results. The same validation partitions selected the historical checkpoints. Bootstrap intervals therefore do not provide independent confirmation, and several contrasts were examined. The result does not establish cross-session or unseen-participant generalization. Missing inputs were evaluated after full-input classifier training; mixed-availability retraining was not tested. No wholly missing window was allowed.

The strongest candidate for an independently confirmed follow-up is the fixed covariance completion approach, keeping both backbones as controls. Such confirmation should use a separately declared independent evaluation protocol. This run does not authorize a new architecture search or tuning the Tucker ranks on the observed validation results.

## Execution and independent review

Each task ran alone under the sequential-session-watchdog workflow, with a 1,200-second process timeout and 15-second kill grace period. The supervisor waited, verified the result, recorded its review and then launched the next task. The A01 pilot was reviewed before cohort execution. Task runtimes were 16.2–23.0 seconds; their sum was 530.0 seconds. All tasks completed without timeouts or retries. No worker or experiment remains running.

Validation included artifact hashes, frozen source/config/package identity, paired deterministic masks, original checkpoint replay, unchanged full-input probabilities, completion state/history and metric recomputation from numeric predictions. The supervisor independently recalculated repeat/seed/participant means, all four paired contrasts and bootstrap intervals; they match the complete report. All 1,411 protected source and historical artifact files remained unchanged.

- [Machine-readable independent audit](real-results-audit.json)
- [Pilot operational review](pilot-review.md)
- [Execution plan](execution-plan.md)
- [Execution receipts](real-execution-receipts.json)
- [Complete scientific report](../../results/completion-transformer-v1/report/summary.md)
- [Per-condition cohort scores](../../results/completion-transformer-v1/report/cohort_conditions.csv)

Full logs and local watchdog code are retained in `.session-runs/completion-transformer/experiments/run-20261005/`. Scientific source and configuration retain the readiness identity `093fd37830cee7228cdc9b8793d7f8919395054f9b23558795b5ce1e26e7f6df`.
