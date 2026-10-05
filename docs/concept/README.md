# Research concept documents

Start with the [current pipeline PDF](current_pipeline.pdf)
([LaTeX source](current_pipeline.tex)) for the baseline research on
`research/baseline-accuracy`: dataset and cue timing, preprocessing and masks,
EEGNet dimensions, cross-entropy, checkpoint selection, and evaluation. It also
explains how the earlier tensor losses and George's separate redesign differ.
This guide describes source revision `4c27ec0`; it introduces no new measurements.

The [spatial model guide PDF](spatial_models.pdf) and
[LaTeX source](spatial_models.tex) cover the completed candidate cohort's
`spatial_eegnet` and `spatial_transformer` models, including exact shapes,
training and validation selection, paired validation findings, and comparison
with published architecture families. The results are validation-only; the
guide explains why they do not establish state of the art or a new architecture.

## Tensor temporal successor

The [tensor temporal guide](tensor_temporal.pdf)
([LaTeX source](tensor_temporal.tex)) describes the successor screen: fixed
spectral Tucker inference, supervised channel–time projection, and factorized
temporal patches, with matched controls and a rerun EEGNet–Transformer reference.
It also distinguishes George's reported test result at `3c9a170` from our
validation measurements. The measured-results fragment is
[`tensor_temporal_results.tex`](tensor_temporal_results.tex).

## Supervised spectral Tucker follow-up

The [supervised Tucker guide](supervised_tucker.pdf)
([LaTeX source](supervised_tucker.tex)) tests classification-trained factors
against matched frozen factors, using identical spectral features, initial
weights, ridge inference and temporal heads. It explains the differentiable
solve, training loss and nine-participant, three-seed evaluation. Measurements
and the verdict are in [`supervised_tucker_results.tex`](supervised_tucker_results.tex).

Run the new study with `bash scripts/run_supervised_tucker_experiments.sh --pilot`,
then `--cohort`. Completed matching fits are verified before reuse; incompatible
or partial artifacts stop execution. Use `--dry-run` to inspect commands and
`--summarize` to verify and aggregate completed evidence.

## Learned covariance completion

The [learned covariance guide](learned_covariance.pdf)
([LaTeX source](learned_covariance.tex)) explains the positive-definite channel
matrix, training-only initialization, regularized observed-channel solve,
gradient flow through the frozen EEGNet–Transformer, and the classification,
reconstruction and anchor losses. It includes vector diagrams, a worked numerical
example, the verified five-arm validation results, and the limits of reused
validation selection. Build it with
`make -C docs/concept learned_covariance.pdf`.

## Earlier frozen-feature concept

[Read the PDF](agfl_concept.pdf) · [Edit the LaTeX source](agfl_concept.tex)

The older document explains the frozen-feature AGFL-inm experiment:
the research question, frozen EEG features, Tucker-2 factors, exact masked ridge
inference, spatial attention, controls, availability masks, and paired evaluation.
Three vector schematics illustrate the pipeline, factorization, and outage schedule.
It distinguishes the original proposal from the supplied implementation and
includes primary references and a dated readiness assessment.

## Rebuild

Build all six Makefile targets from the repository root:

```bash
make -C docs/concept
```

Or build just the current guide from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error current_pipeline.tex
```

Build only the spatial model guide from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error spatial_models.tex
```

Build only the tensor temporal guide from the repository root with:

```bash
make -C docs/concept tensor_temporal.pdf
```

The source uses standard TeX Live packages, including `amsmath`, `booktabs`,
`tabularx`, `tikz`, `hyperref`, and `listings`. It requires neither Python nor
external images, bibliography downloads, or shell escape. References are
contained in the `.tex` file. Build intermediates are ignored locally; the
source and final PDF remain shareable.

## Review after editing

Check each build log for undefined references and overfull boxes. Use `pdfinfo`
to inspect the output and `pdftoppm` to render pages for visual review. Recheck
configuration values and the source revision when research code changes.
Figures are conceptual; the masks are illustrative rather than recorded data.
`current_pipeline.pdf` has no new measurements; `spatial_models.pdf` describes
the completed validation-only candidate cohort with its interpretation limits.
