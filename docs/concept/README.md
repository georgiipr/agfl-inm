# Mathematical concept document

[Read the PDF](agfl_concept.pdf) · [Edit the LaTeX source](agfl_concept.tex)

The document explains the reviewed AGFL-inm experiment for a new collaborator:
the research question, frozen EEG features, Tucker-2 factors, exact masked ridge
inference, spatial attention, controls, availability masks, and paired evaluation.
Three vector schematics illustrate the pipeline, factorization, and outage schedule.
It distinguishes the original proposal from the supplied implementation and
includes primary references and a dated readiness assessment.

## Rebuild

From the repository root:

```bash
make -C docs/concept
```

Or from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error agfl_concept.tex
```

The source uses standard TeX Live packages, including `amsmath`, `booktabs`,
`tabularx`, `tikz`, `hyperref`, and `listings`. It requires neither Python nor
external images, bibliography downloads, or shell escape. References are
contained in the `.tex` file. Build intermediates are ignored locally; the
source and final PDF remain shareable.

## Review after editing

Check the build log for undefined references and overfull boxes. Use `pdfinfo`
to inspect the output and `pdftoppm` to render pages for visual review. Recheck
configuration values and the source revision when research code changes.
Figures are conceptual; the masks are illustrative rather than recorded data.
No classification results or claimed improvements are included.
