# Reproduce the published-checkpoint covariance study

This guide is for a fresh Linux checkout with an NVIDIA GPU. It reproduces the
declared study in a new output directory; the accepted result directory is not
needed and is not included in Git. The published EEG-ATCNet classifier weights
are included under `checkpoints/` (about 6.7 MB total), with SHA-256 identities
recorded in `checkpoints-origin.json`. They are the original frozen weights,
not weights trained by this project. Their upstream source and Apache-2.0
license are identified in [the checkpoint audit](checkpoint-audit.md).

The BCI IV 2a author-format MAT recordings are downloaded by the preparation
script from the dataset's published BNCI Horizon endpoint. The script checks
all 18 files against the hashes used in the completed experiment. Raw GDF data
and the 1.2 GB result bundle are not included. A versioned receipt records the
previous independent MAT-to-GDF and label alignment check; the downloaded MAT
bytes must match that receipt before any fitting begins.

## Requirements

Use Python 3.11, Git, curl, `sha256sum`, GNU `timeout`, a supported NVIDIA
driver/GPU, and enough free disk for the isolated TensorFlow CUDA environment
(about 4.7 GB), downloaded recordings (about 0.8 GB), and fresh results (about
1.2 GB). Actual model fitting requires CUDA. Do not use or modify another
project virtual environment.

From the repository root, install the exact recorded Python package versions
in a new environment:

```bash
python3.11 -m venv .venv-published-covariance
.venv-published-covariance/bin/python -m pip install --upgrade pip
.venv-published-covariance/bin/python -m pip install -r docs/published-covariance/dependencies-lock.txt
```

Prepare the pinned upstream source, checkpoints and author-format recordings:

```bash
bash scripts/reproduce_published_covariance.sh prepare-assets
bash scripts/reproduce_published_covariance.sh plan
```

The plan should declare 27 covariance fits, zero backbone fits and 81 arm
evaluations. Next choose one unused output name, for example
`results/published-covariance-friend-20261020`. Keep using that exact name for
each stage below. Never point these commands at the accepted results or an
existing unrelated study directory.

Run readiness checks before fitting:

```bash
bash scripts/reproduce_published_covariance.sh preflight results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh smoke results/published-covariance-friend-smoke-20261020
```

The native smoke output is a disposable software check, not study evidence.
Choose a new, empty smoke path if you run it again. Review its `native-smoke`
receipt before continuing. The actual experiment is deliberately staged so
the three-task pilot can be reviewed before the other 24 fits:

```bash
bash scripts/reproduce_published_covariance.sh reference results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh pilot results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh review-pilot results/published-covariance-friend-20261020
```

Review the pilot audit, selected epochs, wall time and GPU memory before
proceeding. Do not use pilot accuracy to alter the protocol. Then finish the
cohort and audit it:

```bash
bash scripts/reproduce_published_covariance.sh train-cohort results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh evaluate-cohort results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh audit results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh summarize results/published-covariance-friend-20261020
bash scripts/reproduce_published_covariance.sh verify results/published-covariance-friend-20261020
```

The final audit should require all 27 selected covariance states, nine native
references, 81 arm evaluations and 891 prediction cells. The independent
verifier recomputes the participant-level aggregation and bootstrap interval.
The launcher writes command/output/exit receipts under
`.session-runs/published-checkpoint-covariance/launcher/`.

## What to compare

The accepted run is described in [the verdict](verdict.md) and its numeric
report is in the original run output. Its primary learned-minus-fixed balanced
accuracy difference was 9.55 percentage points (participant-bootstrap 95%
interval 6.32 to 12.55 points). The learned covariance arm scored 77.55% with
16 electrodes and 62.80% with 6; fixed covariance scored 75.46% and 45.79%,
respectively. Small platform differences may alter low-level floating-point
values, so compare the audited protocol, prediction cells and reported
participant hierarchy as well as rounded scores.

The scientific interpretation remains exploratory. The published checkpoints
have grade-B selection provenance, the backbone's historical normalization
used all T trials including the later covariance validation subset, and prior
project exposure to E outcomes is unknown. The three seeds vary covariance
training only; they are not independent pretrained-model seeds. Read the
[checkpoint audit](checkpoint-audit.md), [training specification](training.md)
and [runbook](runbook.md) before interpreting deviations.
