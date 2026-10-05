# Completion + EEGNet Transformer contract

New authorized investigation, 2026-10-05. Preserve all existing scientific code,
configs, results, plans and automation. Work on research/baseline-accuracy.
Implementation belongs only in inm/completion_transformer/, tests/completion_transformer/,
configs/completion-transformer.json and docs/completion-transformer/.
Plans, orchestration and skill files are owned by the supervising agent.

Question: does raw Tucker completion improve missing-input validation accuracy
of the selected EEGNet-front-end temporal Transformer? Is its effect different
from completion on spatial EEGNet? Friend reference is locally available
origin/main at 3c9a170ff22dbe0540053ff278ca9691f866fdf9; do not merge it.

Fixed design: replay the existing reproducible candidate checkpoints, never
retrain neural encoders. Two backbones (spatial_eegnet, spatial_transformer) x
three input strategies (zero, covariance, tucker). Fit completion only on the
original normalized training raw [N,22,4,250], shared across both backbones.
Zero is already normalized training-mean fill. Covariance is a fixed channel
second-moment conditional ridge predictor, ridge 0.001 times mean diagonal;
no per-mask tuning. Tucker reuses inm.tensor_attention.Tucker2 read-only:
ranks 4/16, ridge .001, 30 factor epochs, LR .01, batch 64. Scoped CPU RNG
seed = task seed + 810001; restore ambient RNG. No classifier gradients.

Original Boolean [B,22,4] mask remains the availability feature even when
completed values enter convolution. DO NOT call an existing forward path
which zeroes the filled entries again; DO NOT replace original mask flags with
all-ones. New adapter may explicitly reuse frozen layer modules. Exact zero
and full-input equivalence against original backbone is mandatory. Observed
samples unchanged, hidden NaN/Inf invariance, no all-missing windows. Full input
bypasses completion: no claim of clean accuracy gain is possible here.
Whole-trial EEGNet convolution is retained as a declared successor exception;
completion operates independently per window. It does not recover a wholly
missing window or claim causal online processing.

Nine participants, seeds 0/1/2, historical T-session train/validation only.
Read-only historical inputs results/encoder-candidates-reproducible-v1 and
results/baselines-reproducible-v1. Use verified checkpoint/fit/task hashes,
constructor, split/data IDs, normalization, saved full-input predictions and
relevant historical source/package identities. Fail with a concrete blocker
if replay cannot be verified; do not silently fall back to new training.
New output results/completion-transformer-v1; no writes to historical inputs.
Data/config paths resolve relative to config location, as earlier experiments.
Dependencies may be imported read-only; record all reused source in new identity.
No test scores, checkpoint reselection, architecture search, rank tuning or
source edits after real outcomes. Changed scientific identity needs new output.

Evaluate full_22 and random_static/dynamic_random at 16 and 6 electrodes,
five repeats each degraded condition, paired masks and trial order across all
six cells. Primary: average four degraded-condition BA differences, Tucker
minus zero within Transformer. Secondary: same EEGNet contrast, Tucker minus
covariance within Transformer, and difference-in-differences of the Tucker
benefit between backbones (interaction, not proof of synergy).
Average repeats, then seeds within participant, then participants equally.
Screen: >=2 percentage points primary gain and >=6/9 positive participants;
exploratory participant bootstrap 2000 resamples seed 20261005. Preserve negative
results, per-condition log loss, participant variation and incomplete coverage.
Reused validation selected original checkpoints: not independent confirmation.

Coding sessions use synthetic CPU checks only. Read-only real readiness
verification is permitted; real factor fitting/evaluation is a separate explicit
operation after software review. Never invent real outcomes. Pilot A01/seed0,
then operational review, then cohort only when authorized. No automatic retry.
