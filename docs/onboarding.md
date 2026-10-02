# Collaborator onboarding

## What you are joining

The project asks whether a compact, shared multilinear representation makes
attention-based EEG classification more robust when electrodes disappear.
The supplied study uses four-class motor imagery and nine BCI Competition IV
2a participants. Its main comparison is baseline spatial attention versus
Tucker-core spatial attention, repeated with MHA and Performer and with full
or mixed classifier-training availability. Improvement is a hypothesis.

The tensor stores **electrode × time window × learned feature**. The contribution
being tested is the fitted Tucker factors and inference from observed entries,
rather than storing data in a multidimensional array. Factors capture shared
channel/feature structure; each new trial gets its own inferred core. A fixed
core size does not guarantee that missing information can be recovered.

## Suggested reading order

1. Read [the concept PDF](concept/agfl_concept.pdf) for the mathematical picture.
2. Read [project_overview.md](project_overview.md) and [experiment.md](experiment.md)
   for the implemented scope and declared comparison matrix.
3. Read [model_adaptation.md](model_adaptation.md) and [tensor_math.md](tensor_math.md),
   alongside `inm/model.py` and `inm/tensor_attention.py`.
4. Read [availability_protocol.md](availability_protocol.md), `inm/availability.py`,
   and the original proposal in `references/`.
5. Review [the investigation](investigation.md) before planning experiments.

## Follow a trial through the implementation

| Stage | Files | Shape or responsibility |
|---|---|---|
| Load | `inm/data.py`, `agfl/datasets/base.py` | GDF cues, exclusions, finite trials `[N,22,1000]` |
| Split | `agfl/datasets/splits.py` | Persisted per-class 60/20/20 trial split per participant/seed |
| Normalize | `inm/study.py::_calibrate` | Raw channel statistics fitted on training trials |
| Pretrain | `inm/model.py`, `inm/training.py` | Shared window-local encoder with a full-input MHA head |
| Freeze and standardize | `inm/study.py::_calibrate` | Learned features `[N,22,4,32]`, training-only feature statistics |
| Fit representation | `inm/tensor_attention.py` | Frozen factors `U[22,4]`, `V[32,4]`; trial core `[4,4,4]` |
| Apply availability | `inm/availability.py` | Boolean mask `[N,22,4]`, deterministic partition-disjoint subsets |
| Classify | `inm/model.py::FeatureClassifier` | 22 electrode tokens or 4 latent tokens per window; four logits |
| Evaluate | `inm/study.py`, `inm/training.py` | Fixed validation-selected heads, full and degraded conditions |
| Aggregate | `inm/reporting.py`, `inm/plots.py` | Paired comparisons and optional figures from saved tables |

The other `agfl/models/` architectures are library components. The availability
study uses its own EEGNet-derived channel-local encoder, not the standard
full-trial EEGNet or Signal Transformer backbone.

## The comparisons you should be able to explain

| Representation | Interpretation | Attention |
|---|---|---|
| Baseline | Observed learned features; learned missing placeholders | MHA, Performer |
| Tucker core | Masked ridge-inferred latent channel/feature components | MHA, Performer |
| Tucker completion | Keep observations; fill only missing features from the factor model | MHA |
| Linear core | Supervised separable compression, matching latent shape | MHA |
| Mean completion | Missing channel-feature values filled with training mean | MHA |

Every row runs under full and mixed downstream training. There are 14 heads
per participant/seed, 27 such tasks, 378 head fits, plus 27 shared encoder fits
and 27 Tucker calibrations. Calibration uses full-channel training inputs in
both regimes. Thus “mixed training” describes the downstream classifier.

Attention runs independently within each of four windows, and window readouts
are averaged. There is no temporal attention or time-position embedding. A joint
permutation of feature and mask windows leaves the evaluation classifier unchanged.

## Commands and prerequisites

From the repository root, inspect the complete task plan without loading EEG:

```bash
python run.py --config configs/study.json --plan
```

Actual training needs Python 3.12+, the packages in `requirements.txt`, a
compatible CUDA PyTorch installation, and the nine T-session GDF files. Use the
[repository README](../README.md) for acquisition and environment instructions.
The default data location `../ml` is resolved from the launching directory.
An external Slurm launcher is mentioned in the README but is not supplied here.

Once prerequisites are ready, task 0 is A01, seed 0, and runs all 14 heads:

```bash
python run.py --config configs/study.json --task-index 0
```

After results exist, rebuild saved reports and optionally figures:

```bash
python run.py --config configs/study.json --summarize-only
python run.py --config configs/study.json --summarize-only --plots
```

Start reading results at `results/inm-v2/report/summary.md`. Retain `artifacts/`
for splits, normalization, encoder/factors, histories, and provenance. Selected
head weights are evaluated in memory and are not saved as reusable checkpoints.
If deployment is a goal, checkpoint persistence requires a separate change.

## A productive first contribution

For the separate baseline-first accuracy study, use the
[baseline runbook](baselines/runbook.md) for its environment, paths, smoke,
pilot, checkpoints, and reporting. Its model is an architecture reproduction,
not by itself a paper-protocol reproduction; its benchmark scores remain
unknown until the real experiment is run.

Discuss the target protocol with the supervisor: the implemented participant-specific
study, the original subject-disjoint proposal, or a later cross-session study.
These answer different questions and should receive distinct study identities.
Then verify the dataset and cluster environment and run one participant/seed
task before requesting the full allocation. Check trial exclusions, selected
epochs, factor history, parameter counts, and matching mask hashes.

Useful implementation checks for future legacy-experiment work include agreement
of the core solve with an explicit ridge system, invariance to hidden NaNs,
preservation of observed features in completion, partition-disjoint masks, and
reporting of incomplete pairs. The separate baseline package has a CPU unittest
suite under `tests/baselines/`; see its [runbook](baselines/runbook.md) and
[integration review](baselines/review.md). Those baseline checks do not imply
that the legacy experiment has the same automated coverage.

For interpretation, compare both degraded and full-input balanced accuracy,
inspect participant variation, and use the completion/linear/mean controls.
Avoid attributing a gain solely to tensor structure when token counts, pooling,
trainable capacity, or MHA-based feature pretraining could explain it.
