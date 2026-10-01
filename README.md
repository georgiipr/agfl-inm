# AGFL-inm: EEGNet features, tensor attention and missing EEG channels

This is a separate research project built from the existing AGFL codebase and
`Proposal_Tensor_Attention_Variable_Signal_Availability`. It measures whether
training-fitted Tucker-2 representations improve classification when electrode
availability changes. It does **not** assume an accuracy improvement.

All nine BCI Competition IV 2a participants are trained **individually**. The
supplied Slurm job performs the complete experiment; no tests or training were
run while preparing this project. The original AGFL project is not modified.

The maintained model library contains **EEGNet** and **Signal Transformer**, with
**MHA** and **Performer** selected independently. The supplied experiment still
uses the EEGNet-derived window encoder; Signal Transformer is retained as a model
implementation and is not added to this study's matrix.

## Start here

- [Project goals and proposal mapping](docs/project_overview.md).
- [Mathematical apparatus and Tucker-2 API](docs/tensor_math.md).
- [Standalone tensor/hypermatrix implementation](inm/tensor_attention.py).
- [EEGNet adaptation and representation controls](docs/model_adaptation.md).
- [Channel-availability protocol](docs/availability_protocol.md).
- [Experimental protocol, outputs and interpretation](docs/experiment.md).
- [Complete configuration](configs/study.json) and [Slurm launcher](run_inm.sbatch).

“Tensor” and “hypermatrix” mean the same multidimensional array here. The tested
intervention is the learned Tucker-2 factorization and masked core inference,
not two unrelated mechanisms bearing those names.

## What runs

| Setting | Default |
|---|---|
| Participants | A01–A09, trained separately |
| Seeds | 0, 1, 2 |
| Input | 22 EEG channels; four 1-second windows per cue trial |
| Dataset files | `../ml/A01T.gdf` through `../ml/A09T.gdf` |
| Split | Within each participant's T session: stratified 60% train / 20% validation / 20% test |
| Backbone | Shared, frozen EEGNet-derived channel-local window encoder |
| Attention | MHA, Performer |
| Primary routes | Baseline and Tucker-2 core for each attention |
| MHA controls | Tucker completion, separable linear core, training-mean completion |
| Classifier availability training | Full channels; mixed channel availability |
| Evaluation | Full channels; static random/spatial and dynamic random/spatial losses |
| Channels retained | 22, 16, 11, 6 |
| Mask repeats | Five per degraded condition; one full-input condition |

There are **14 classifier fits per subject/seed, 378 classifier fits total**,
plus 27 shared encoder fits and 27 factor fits. Each selected classifier is
evaluated on 13 conditions on both validation and held-out test data; it is
**not retrained for every evaluation mask**. Six retained electrodes means
72.73% exclusion, the largest whole-channel removal that does not exceed 75%.

The encoder and factors first receive full-channel **training calibration**.
`full` and `mixed` describe subsequent classifier training. Attention operates
spatially within each window; there is no temporal attention for EEG. Core
tokens are latent channel components, not named brain regions.

The old temporal-attention studies, replay branches, presets and plotting
commands have been removed. Temporal convolutions still encode each EEG window,
and dynamic channel availability still changes between windows. These operations
do not introduce attention across time. `run.py` is the experiment entry point;
`agfl/` now supplies library code rather than a second experiment CLI.

This reduced matrix uses schema version 2 and `results/inm-v2`. Earlier source,
configuration and provenance manifests are preserved in
`references/pre-cleanup-v1.tar.gz`. Existing results are not deleted or relabeled.

## Upload from the Mac

Run this in your Mac terminal after stopping writers using this project's source.
The first transfer removes retired files **only inside this project's `agfl/`
source directory**. The second uploads the rest without deleting cluster results:

```bash
rsync -av --delete --progress /Users/egor/Downloads/AGFL-inm/agfl/ \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL-inm/agfl/

rsync -av --progress --exclude='results/' --exclude='*.log' \
  --exclude='.venv/' --exclude='__pycache__/' \
  /Users/egor/Downloads/AGFL-inm/ \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL-inm/
```

The destination is a sibling of your existing `AGFL` and `ml` folders. The
project is self-contained; it does not import from the other AGFL checkout.
It uses the existing Python **3.12+** environment and existing AGFL dependencies.
No package installation, pytest invocation or preflight training is in the job.
`requirements.txt` records the already-used packages, not a new install step.

## Submit on the cluster

Working directory: `/trinity/home/georgii.promyslov/AGFL-inm`.
Dataset directory: `/trinity/home/georgii.promyslov/ml` by default.
Output directory: `AGFL-inm/results/inm-v2`.

```bash
cd /trinity/home/georgii.promyslov/AGFL-inm
sbatch run_inm.sbatch
```

The array has tasks 0–26, at most two running concurrently. Each task requests
one GPU, four CPUs, 16 GB RAM and 24 hours. Task `3*(subject-1)+seed` runs that
participant/seed combination and its 14 classifiers. Actual duration is unknown
until cluster execution; the time limit is a resource request, not a prediction.
The launcher activates `$HOME/.venv`. If needed, set `AGFL_INM_VENV` to the existing
environment directory before submission. Edit `data.data_dir` in
`configs/study.json` before submission if the GDF folder is elsewhere.

For one concurrent GPU allocation instead, submit the same experiment as:

```bash
sbatch --array=0-26%1 run_inm.sbatch
```

This changes concurrency only. Do not submit both commands for the same output
folder simultaneously. For a longer permitted allocation, an `sbatch --time=...`
override avoids changing the source file while a study is in progress.

## Progress and interrupted jobs

`sbatch` prints the job ID. For example, replace `JOBID` below with that number:

```bash
squeue -j JOBID
tail -n 40 agfl-inm-JOBID_0.log
```

Logs show stage/fit transitions and full-input test accuracy after each completed
fit. Epoch progress uses a progress bar when a terminal is available; batch logs
avoid per-epoch console spam. Epoch histories remain in `artifacts/`.

Resubmitting `sbatch run_inm.sbatch` reuses **only completed fits with matching
code, configuration, environment, input-data, split and calibration identities**.
An interrupted classifier fit restarts; completed fits are retained. To resubmit
only task 8, for example:

```bash
sbatch --array=8 run_inm.sbatch
```

Do not modify code/configuration during a running study. Changed scientific code,
packages or settings require a new `output_dir`, such as `results/inm-v3`; old
scores will not silently count toward the new experiment. Documentation-only
edits do not change its identity.

## Accuracy tables and diagnostics

Reports refresh automatically after fits. Download only `report/`, which contains
the tables, progress/errors, data summary and source/configuration manifest:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL-inm/results/inm-v2/report/ \
  /Users/egor/Downloads/AGFL-inm-report/
```

Open `summary.md` first. `accuracy_by_run.csv` shows completed runs even while
other jobs are pending. `accuracy_table.csv` contains nine-subject averages;
these stay blank until all required subjects/seeds are complete.
`paired_tensor_gain.csv` answers whether Tucker improves the same attention
under matched conditions. See [output definitions](docs/experiment.md).

If you want to rebuild the saved-results summary on the cluster, no GPU job is
needed:

```bash
cd /trinity/home/georgii.promyslov/AGFL-inm
source "$HOME/.venv/bin/activate"
python run.py --config configs/study.json --summarize-only
```

This reads saved results; it does not train or evaluate models. There is no
separate checkpoint-diagnostic command: selected classifier weights stay in
memory, all declared evaluation conditions are scored automatically, and the
stored per-class metrics/errors/histories provide the diagnostics. Shared encoder,
features, normalization and tensor-factor state are retained in `artifacts/`
for provenance and restart, separate from the downloadable report.

## Optional plots — separate command

After the needed groups have completed, use the already available AGFL plotting
environment on the cluster:

```bash
cd /trinity/home/georgii.promyslov/AGFL-inm
source "$HOME/.venv/bin/activate"
python run.py --config configs/study.json --summarize-only --plots
```

This uses the existing optional Matplotlib dependency; nothing installs a library.
It writes PNG/PDF curves and paired-gain figures to `report/plots/` and a
`report/plots_status.json` manifest. Incomplete figure groups are skipped with
an explicit reason. The main `.sbatch` run needs no plotting dependency.
Repeat the separate download command above to fetch changed reports and plots.

## Verification status

Preparation includes static Python/configuration/shell checks and code review.
No project imports, tests, model inference, training, Slurm submission or numerical
validation were executed locally. Runtime behavior and measured gains remain
unverified until you run the cluster experiment.
