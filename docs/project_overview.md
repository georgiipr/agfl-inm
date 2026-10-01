# Project overview and research goals

## Research question

Does an explicit, training-fitted multilinear representation preserve useful
classification information when known EEG channels disappear, compared with
the same attention family receiving masked EEGNet-derived features directly?

The primary hypothesis is improved **balanced accuracy under channel loss**.
Full-input accuracy is a separate outcome: compressing or completing features
may harm it. The experiment must report both improvements and regressions.
We also ask whether a benefit comes from completion, compression, or exposing
the classifier to missing inputs during training.

## Pipeline

1. Load a single participant's T-session trials with four motor-imagery classes.
   Exclude declared artifacts and unusable/out-of-bounds epochs. Split trials
   into training, validation and held-out test partitions before fitting statistics.
2. Filter each channel's disjoint one-second window independently. Fit channel
   normalization on training samples only.
3. Fit a shared EEGNet-derived channel-local encoder with a spatial MHA head.
   Select its epoch using full-input validation. Freeze it and discard that head.
4. Compute a feature tensor `X[trial,22,4,32]`; standardize channel-feature values
   using training trials only. Fit one shared Tucker-2 factor pair using training
   features, without labels. Freeze it.
5. Train the declared classifier heads with either full or mixed availability.
   Each receives only observed feature entries and the mask. Both routes use
   the same encoder, splits, budgets and availability schedules.
6. Select heads using full-input validation; evaluate the fixed heads at all
   declared availability conditions on validation and held-out test partitions.
7. Average repeated masks, seeds within each participant, and then participants
   equally. Compare paired tensor-minus-baseline differences. Preserve negative
   results, failures and missing cells visibly.

Full feature caches may contain synthetic hidden reference values. This is safe
for the declared simulation because the frozen encoder is channel/window-local,
and masks remove those values **before any classifier or tensor inference**.
Only the reconstruction scorer accesses hidden feature values as targets.
This does not simulate training an encoder from permanently absent channels.

## Proposal mapping

The supplied PDF and same-name DOCX are preserved in `references/`; the readable
extraction was made from the DOCX. `references/upstream_snapshot.json` records
the upstream identities of retained source files. The full pre-cleanup source,
configuration, documentation and hash manifests are preserved in
`references/pre-cleanup-v1.tar.gz`. The EEG experiment lives in `inm/`.

| Proposal element | This implementation |
|---|---|
| Tensor/hypermatrix `channels × windows × features` | Implemented with explicit original channel IDs and Boolean masks |
| Shared Tucker-2 channel/feature factors | Implemented; frozen train-only fit with exact masked ridge core solves |
| Core and observed-preserving completion routes | Both implemented |
| Masked reconstruction objective | Implemented in standalone tensor module; this study calibrates once on full training inputs |
| Fixed spectral features | Adapted to frozen EEGNet-derived window features to satisfy the requested backbone |
| Subject-disjoint folds | Replaced, at the user's direction, by individual training for all nine subjects |
| Temporal position embeddings | Not included: spatial attention per window, followed by order-invariant window averaging |
| Baseline attention comparisons | MHA and Performer |
| Bottleneck/mean-imputation controls | MHA separable linear core and training-mean completion |
| Availability varying between training/evaluation | Full vs mixed classifier-training regimes; held-out masks/count/patterns |

These are explicit adaptations, not an exact reproduction of the proposal or
standard EEGNet. In particular, changing window filtering and freezing an adapted
encoder means the absolute scores cannot be compared directly with earlier
AGFL/EEGNet experiments as though only the attention changed.

## What this first experiment can establish

- Whether frozen Tucker core inference helps each attention under the specified
  channel losses, and at what cost on full inputs.
- Whether feature completion or a supervised separable linear bottleneck explains
  similar behavior in the MHA controls.
- Whether mixed-availability classifier training reduces degradation on unseen
  11-channel inputs and spatial outage patterns.
- How effects vary across all nine participants, rather than selecting only
  favorable participants after seeing results.

It cannot establish improvements on every dataset/backbone, generalization to
unseen sensor identities or participants, true anatomical connectivity, or an
end-to-end standard EEGNet gain. Positive confidence bounds in this exploratory
matrix are not a multiple-comparison-corrected confirmatory claim.

## Implementation map

| File | Role |
|---|---|
| `run.py` | Cluster entry point and saved-report commands |
| `inm/tensor_attention.py` | All Tucker/hypermatrix fitting and inference mathematics |
| `inm/model.py` | EEGNet-derived features, spatial heads, representation controls |
| `inm/availability.py` | Deterministic static/dynamic masks and partition-disjoint banks |
| `inm/data.py` | BCI2a loading and filtering restricted to each window |
| `inm/training.py` | Shared training and validation-only epoch selection |
| `inm/study.py` | Subject tasks, calibration, paired fits, evaluation and provenance |
| `inm/reporting.py` | Complete/partial tables and subject-level paired intervals |
| `inm/plots.py` | Optional figures from saved tables, no checkpoint inference |
| `agfl/` | Maintained EEG library: two models, two attentions, data/split and optimization helpers |

The original project is not needed at runtime. Experimental changes can be
removed by setting aside this separate folder; the original code is untouched.

## Maintained scope

`agfl/models/` exposes exactly `eegnet` and `signal_transformer` through
`get_model_spec`. Its `build(model_options, metadata, attention=...,
attention_options=...)` API accepts EEG metadata and independently selects
`mha` or `performer`. Both models attend across electrodes. EEGNet retains
compact and spatial-fusion readouts; Signal Transformer retains its shared
per-electrode temporal tokenizer and residual spatial-attention blocks.

The availability experiment has its own channel/window-local encoder in
`inm/model.py`. It remains EEGNet-derived: adding Signal Transformer to this
experiment would require an explicit window-local adaptation and a new declared
comparison. The retained full-trial model library is not used as a masked encoder.

The former generic training CLI, tuning, study-specific analysis, checkpoint
replay, presets and visualization tree have been removed from active source.
`inm/reporting.py` and `inm/plots.py` own the current reports and plots.
Temporal kernels and dynamic masks remain; temporal-attention workflows do not.

The pre-cleanup archive contains the former source tree, study settings,
documentation and complete upstream/project hash manifests. It is provenance,
not an importable package or a second active protocol. The current project hash
manifest covers the maintained files and the archive checksum. The original
proposal documents remain preserved separately.
