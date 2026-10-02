# Research branch and collaborator review

Reviewed on 2026-10-02. Continue the baseline accuracy investigation on
`research/baseline-accuracy`. George's `origin/main` currently ends at
`92b7fc2ba8ca360777d7c97d9ed961d24b67ccaf` and implements a different experiment.
The most useful research idea is raw-signal completion before a strong classifier.
First finish the existing baseline training/generalization audit; the new branch
does not provide evidence that a more complex classifier improves accuracy.

## Repository organization

The common starting revision is `84ae174`. Commit `9938aa4` preserves the existing
local research code, configurations, tests, plans, documentation, and result
reviews, including `AGENTS.md`. This is a snapshot of prior work, not a claim
that all of it was implemented during this review. Recordings, environments,
session logs, and generated experiment results remain outside Git.

George contributed two subsequent commits:

| Commit | Change | Decision for this branch |
|---|---|---|
| `b9adef2` | Anchor recording-directory ignore patterns to the repository root | Adopted as `796d72a`, preserving George's authorship; source under `agfl/datasets/` is now visible to ordinary file searches |
| `92b7fc2` | Replace the frozen-feature experiment with four end-to-end models | Reviewed separately; retain as a reference for a future bounded ablation |

The research branch retains the legacy 27-task/378-fit plan and the separate
baseline package. George's rewrite renames `agfl/` to `eeg_models/`, removes
`inm/model.py`, moves Tucker code, replaces the protocol with schema 3, and
removes the proposal files. A wholesale merge would require a deliberate
migration of baseline imports and provenance, and would change the scientific
comparison. Historical result directories must retain their original identities.

Continue work from this checkout with `git switch research/baseline-accuracy`.
To inspect future contributions without updating this branch's files:

```bash
git fetch origin
git log --oneline 92b7fc2..origin/main
git diff 92b7fc2..origin/main
```

Review individual changes before cherry-picking them. Keep this branch's remote
destination separate from `main` if it is published later. This organization
created local commits only; no remote branch was published.

## What George changed scientifically

Source links below are pinned to the reviewed commit, rather than a moving branch.

| Dimension | Preserved research | George's new experiment |
|---|---|---|
| Main comparison | Separate spatial EEGNet/covariance baselines; frozen-feature tensor follow-up | EEGNet and Signal Transformer, each with and without raw-signal Tucker completion |
| Tensor input | Frozen learned features `[B,22,4,32]` in the legacy pipeline | Normalized raw windows `[B,22,4,250]`, default ranks 4 and 16 |
| Classifier training | Full and mixed availability; validation policy declared per study | Full-input training only; full-input validation selects epochs |
| Attention | No attention required in the accuracy baseline; MHA/Performer in legacy | Fixed four-head MHA in both backbones |
| Temporal behavior | Flatten baseline retains ordered time bins; legacy averages window readouts | Whole-trial encoding after masking/completion, retaining ordered temporal information |
| Default budget | Baseline: 108 fits; legacy: 378 heads plus calibration | 108 model fits across 27 participant/seed tasks |

These are source and configuration differences, not measured improvements.
The tracked tree at `92b7fc2` contains no test suite or measured result artifacts.
Both branches' default experiments remain within-participant T-session studies;
the rewrite does not establish cross-session or unseen-participant performance.

## Useful ideas and interpretation limits

**Complete raw signals before encoding.** The new
[SignalInput](https://github.com/georgiipr/agfl-inm/blob/92b7fc2ba8ca360777d7c97d9ed961d24b67ccaf/eeg_models/models/_shared/input.py)
selects observations with `torch.where`, then fills only missing entries using
training-fitted factors. This lets a classifier learn spatial structure end to
end. It is a different hypothesis from compressing weak frozen features, so the
negative frozen-core findings do not rule it out. Its accuracy benefit is unknown.

**Use one trained backbone to isolate completion at inference.** The new
[runner](https://github.com/georgiipr/agfl-inm/blob/92b7fc2ba8ca360777d7c97d9ed961d24b67ccaf/inm/study.py)
resets initialization before each model. Completion adds buffers and is bypassed
on full-input batches. Synthetic checks confirmed identical full-input logits
and parameter gradients for both baseline/tensor pairs under matched seeds.
Consequently, matched deterministic training should produce the same backbone
within each pair. Tensor completion cannot improve full-input predictions for
that shared backbone because it is an exact pass-through. The experimental
question is its effect on missing-input predictions. A future implementation
could fit each backbone once and evaluate zero-fill versus completion using the
same selected checkpoint, after verifying state and prediction equality.

**Keep strong simple controls.** George's
[EEGNet](https://github.com/georgiipr/agfl-inm/blob/92b7fc2ba8ca360777d7c97d9ed961d24b67ccaf/eeg_models/models/eegnet/backbone.py)
adds electrode MHA at every temporal sample, before spatial convolution. It is
not a plain EEGNet control. Adding that MHA and raw completion simultaneously
would not isolate which changed the outcome. Our existing end-to-end EEGNet
already provides a useful simpler starting point. Profile runtime before
adopting per-sample MHA; this review measured no throughput improvement.

**Describe exactly how masks affect attention.** In EEGNet, missing electrodes
are excluded as keys at each sample, but their queries and residual activations
still feed spatial convolution. In
[Signal Transformer](https://github.com/georgiipr/agfl-inm/blob/92b7fc2ba8ca360777d7c97d9ed961d24b67ccaf/eeg_models/models/signal_transformer/backbone.py),
an electrode is an eligible key if any window was observed. Its token summarizes
the whole trial; this is not attention with a separate mask per window. Completed
values can therefore affect the network despite retaining the original key mask.
These choices need explicit controls if adapted to our study.

**Reuse selected checkpoints after interrupted evaluation.** George's runner
can restore a selected checkpoint and continue the availability sweep. Our
baseline already saves and verifies reloadable checkpoints, but this separation
of fitting from evaluation is useful future maintenance work. Port the behavior
through the baseline's existing identity and checkpoint contracts rather than
copying the incompatible schema-3 runner.

## Next bounded research task

Follow the recommendation in the
[completed tensor review](tensor-results-review.md): audit the selected
reproducible spatial EEGNet checkpoints on clean training and validation trials,
including per-class behavior, participant variation, cue alignment, and the
declared preprocessing. The [corrected review](corrected-results-review.md)
records 91.28% clean training versus 60.86% validation balanced accuracy for the
mask-conditioned full-training baseline. This gap motivates diagnosis before
adding capacity. These numbers are cited from the existing review; they were
not recomputed here.

If that audit supports a completion experiment, predeclare a small comparison
of zero-fill, a simple training-fitted completion control, and raw Tucker
completion with the same selected EEGNet, splits, seeds, and masks. Normalized
zero-fill already represents the per-channel training mean, so another such
mean-fill arm would duplicate it. Use a new output identity and choose settings
only on the declared validation conditions; keep test and 11-channel/spatial-loss
outcomes out of selection. Introduce MHA or Signal Transformer as a later,
separate question. Full-only training tests robustness to unseen loss; mixed
training tests a distinct training regime and must be reported separately.

## Checks performed

- The preserved baseline suite passed all 65 tests on CPU.
- Legacy planning still reports 27 tasks and 378 classifier fits; the
  reproducible baseline planner reports 27 tasks and 108 fits.
- George's archived source planner reports 27 tasks and 108 model runs.
- In a temporary source archive, synthetic checks covered all four new models:
  finite degraded outputs and parameter gradients with hidden NaNs, unchanged
  observed samples, frozen factor buffers, and equality of full-input logits and
  parameter gradients within each matched backbone pair. Factors were fitted for
  one epoch on synthetic data solely for this check.
- No real-data training, complete training-trajectory equality check, CUDA
  performance benchmark, or empirical evaluation of George's models was run.

The preserved snapshot contains a pre-existing extra blank line at the end of
`inm/baselines/__init__.py`. It was left intact to preserve the source bytes tied
to completed studies. This review introduces no Python or configuration changes.
