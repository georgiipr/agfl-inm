# AGFL-inm

EEG classification under changing channel availability, comparing spatial
**MHA** and **Performer** with baseline features and training-fitted **Tucker-2**
representations.

The model library contains **EEGNet** and **Signal Transformer**, with attention
selected separately. The supplied experiment uses a shared, frozen
EEGNet-derived window encoder; Signal Transformer is available in the library
but is not part of this experiment matrix. Attention operates across electrodes
or latent spatial components within each window. Temporal convolutions and
dynamic availability masks are supported; attention is not applied across time.

## Get the project

```bash
git clone https://github.com/georgiipr/agfl-inm.git
cd agfl-inm
```

Run the commands below from the repository root on the experiment machine.

## Python environment

Use **Python 3.12 or newer**, a CUDA-capable GPU, and the packages listed in
[requirements.txt](requirements.txt). On a cluster, reuse an existing compatible
Python virtual environment with CUDA-enabled PyTorch:

```bash
export AGFL_INM_VENV=/path/to/your/venv
source "$AGFL_INM_VENV/bin/activate"
```

If you need to provision a separate environment, install the listed dependencies
in a virtual environment on the experiment machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Workstation CUDA build; choose a compatible build for other machines.
python -m pip install 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
export AGFL_INM_VENV="$PWD/.venv"
```

Use a PyTorch build compatible with the cluster's GPU drivers. Optional plots
also require Matplotlib 3.8 or newer, below version 4. The external Slurm launcher installs
nothing and does not require pytest. It activates `AGFL_INM_VENV`, falling back
to `$HOME/.venv` if that variable is unset.

For the baseline study, `bash scripts/run_baselines.sh --check-cuda` selects the
project `.venv` (or `AGFL_PYTHON`) and verifies a GPU forward/backward computation.
Run it from a terminal with GPU access; a sandbox's CUDA failure does not prove
the host GPU is unavailable. Real baseline tasks default to CUDA:

```bash
bash scripts/run_baselines.sh --config configs/baselines.json --preflight
bash scripts/run_baselines.sh --config configs/baselines.json --task-index 0
```

The coding-session runner also selects the project `.venv` by default. Its
acceptance tests and baseline `--smoke` continue to use CPU. See the
[baseline environment guide](docs/baselines/environment.md) for overrides.

## Dataset: BCI Competition IV, data set 2a

The dataset contains nine participants performing four motor-imagery tasks:
left hand, right hand, both feet, and tongue. Recordings contain 22 EEG channels
at 250 Hz, plus three EOG channels that this project excludes from classifier
inputs. See the [official dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf).

1. Open the [official competition download page](https://www.bbci.de/competition/iv/#download),
   review the dataset terms and citation requirements, and follow the download
   link for **Data sets 2a: GDF files zipped**.
2. Extract the archive. This configuration requires the nine training-session
   recordings named **`A01T.gdf` through `A09T.gdf`**, with their original names.
3. Put those files directly in a directory named `ml` beside the repository:

   ```text
   workspace/
   ├── agfl-inm/
   │   ├── run.py
   │   └── configs/study.json
   └── ml/
       ├── A01T.gdf
       ├── A02T.gdf
       ├── ...
       └── A09T.gdf
   ```

   Create the directory with `mkdir -p ../ml` from the repository root, then
   copy the extracted GDF files into it. Keep recordings outside version control.
4. For a different location, set `data.data_dir` in
   [configs/study.json](configs/study.json) before submission. The default is
   `../ml`; relative paths are resolved from the directory where you launch.

Use the GDF release rather than a converted MATLAB/NPZ dataset. Labels for the
T-session files are read from their cue events. The supplied study does not need
`A01E.gdf`–`A09E.gdf` or separate evaluation-label files.

**Evaluation protocol:** each participant's T session is split into approximately
60% training, 20% validation and 20% test trials, stratified by class. This is a
within-session, participant-specific experiment; it does not implement the
competition's train-on-T/test-on-E benchmark. Training statistics and Tucker
factors use training data only; full-input validation selects epochs.

## Experiment configuration

| Setting | Supplied configuration |
|---|---|
| Participants and seeds | A01–A09 trained individually; seeds 0, 1, 2 |
| Input | 22 EEG channels; four 250-sample windows per trial |
| Encoder | Shared, frozen EEGNet-derived window encoder |
| Attention | MHA and Performer |
| Main representations | Baseline and Tucker-2 core |
| Additional MHA controls | Tensor completion, separable linear core, mean completion |
| Classifier training | Full availability and mixed availability |
| Evaluation | Full input; static/dynamic random and spatial channel loss |
| Retained channels | 22, 16, 11, 6 |
| Repeated masks | Five per degraded condition; one full-input condition |
| Output directory | `results/inm-v2` |

There are **27 participant/seed tasks**, each fitting one shared encoder, one
Tucker factor pair, and **14 classifiers**: **378 classifier fits** in total.
Each selected classifier is evaluated across all declared availability conditions
without retraining for individual masks. Configuration details and interpretation
limits are in [the experiment protocol](docs/experiment.md).

## Reports and plots

Reports refresh after classifier fits. Start with
`results/inm-v2/report/summary.md`. Individual completed runs appear in
`accuracy_by_run.csv`; all-participant averages remain blank until their required
fits are complete. `paired_tensor_gain.csv` contains matched tensor-minus-baseline
comparisons.

Rebuild reports from saved results on the experiment machine:

```bash
python run.py --config configs/study.json --summarize-only
```

Generate the optional PNG/PDF availability curves, MHA-control comparisons, and
paired-gain plots:

```bash
python run.py --config configs/study.json --summarize-only --plots
```

Figures are saved under `results/inm-v2/report/plots/`. The accompanying
`plots_status.json` explains any figure groups skipped because results are
incomplete. These commands read saved results; they do not retrain models.
The plotting environment must already contain Matplotlib.

To copy reports from a cluster, replace the account, host and remote project
location in this example:

```bash
rsync -av --progress \
  USER@HOST:/path/to/agfl-inm/results/inm-v2/report/ \
  ./downloaded-report/
```

Keep `artifacts/` with the study for calibration state, splits, histories and
per-fit records. Selected classifier weights are evaluated in memory and are
not saved as separate checkpoints. Reports preserve unfavorable results and
explicitly mark incomplete comparisons.

## Project map

- [Accuracy improvement sessions](plans/accuracy/README.md): bounded implementation plans
  and a sequential, resumable Codex runner for smaller models.
- [Collaborator onboarding](docs/onboarding.md): guided reading and practical first steps.
- [Concept PDF](docs/concept/agfl_concept.pdf): mathematical explanation and vector schematics;
  [LaTeX and build instructions](docs/concept/README.md).
- [Repository investigation](docs/investigation.md): reviewed evidence, readiness, and limitations.
- [Documentation index](docs/README.md): all orientation and technical documents.
- [Project overview](docs/project_overview.md): research question and implementation map.
- [Model adaptation](docs/model_adaptation.md): encoder, representations and masking.
- [Tensor mathematics](docs/tensor_math.md): Tucker-2 fitting and masked core inference.
- [Availability protocol](docs/availability_protocol.md): deterministic outage masks.
- [Experiment protocol](docs/experiment.md): splits, selection, aggregation and outputs.
- `agfl/`: EEG model, attention, data and optimization helpers.
- `inm/`: experiment implementation, reports and optional plots.
- `references/`: research proposal documents.
