# AGFL-inm

EEG classification with **EEGNet** and **Signal Transformer**, each with and
without training-fitted tensor completion. **MHA is built into both backbones**,
which train end to end.

## Goals

Study whether tensor completion improves existing EEG classifiers as channel
availability decreases, using matched baselines and a reproducible evaluation
protocol.

1. Establish full 22-channel performance for both existing backbones with MHA.
2. Compare each backbone with its tensor version using the same trials, splits,
   initialization seeds, preprocessing and availability masks.
3. Measure accuracy, stability and degradation as channels decrease to 16, 11
   and 6, including static loss and channels disappearing and returning over time.
4. Report all nine participants with equal weight, including unfavorable or
   inconclusive tensor results. Better reconstruction alone is not evidence of
   better classification.

## Four models

| Model key | Implementation |
|---|---|
| `eegnet` | Whole-trial EEGNet: temporal convolution → electrode MHA → full-channel depthwise spatial convolution → separable convolution → classifier |
| `eegnet_tensor` | The same EEGNet, preceded by Tucker-2 completion of missing signal windows |
| `signal_transformer` | Per-electrode temporal tokenizer → residual spatial MHA blocks → learned spatial readout → classifier |
| `signal_transformer_tensor` | The same Signal Transformer, preceded by Tucker-2 completion |

Both backbones train end to end. EEGNet retains its full spatial filter and
ordered pooled time bins; Signal Transformer retains ordered temporal bins in
its tokenizer. MHA always has four heads and mixes electrodes: at each time
sample in EEGNet, and over whole-trial electrode tokens in Signal Transformer.
Temporal processing and dynamic outages remain part of the workflow.

The tensor input is `trial × electrode × window × within-window sample`
(default `B × 22 × 4 × 250`). Channel and within-window factors are fitted on
normalized **training data only**, once per participant/seed, and then frozen.
Completion preserves measured samples exactly and fills only missing entries
before the backbone. With all 22 channels present it is an exact pass-through.
The original availability mask still controls MHA keys in both variants; sensor
positions never shift when channels disappear. Tensor completion preserves the
backbone input shape.

## Experiment pipeline

For **each participant and seed**:

1. Load that participant's T-session trials and make a fixed stratified split:
   approximately 60% train, 20% validation, 20% held-out test.
2. Fit channel normalization on training trials. Fit shared Tucker factors on
   full-channel training windows if any tensor model is selected.
3. For each selected model, train the entire model **once on all 22 channels**.
   Only 22-channel validation runs during training and selects the checkpoint
   by balanced accuracy, with validation log loss breaking ties.
4. Freeze the selected checkpoint. Evaluate **22 → 16 → 11 → 6 channels** on
   validation, then held-out test, without further fitting or retraining.
5. Record all scenarios and repeats, selected weights, training history, split,
   preprocessing/factor state, source hashes and configuration provenance.
6. Average mask repeats, then seeds within each participant, then give each of
   the nine participants equal weight. Report participant variation and paired
   tensor-minus-baseline gains. Incomplete averages remain blank.

Degraded validation is descriptive after selection. It does not select epochs,
ranks or settings; held-out test scores never influence those decisions.
Each decreased count has static random, static spatial, dynamic random and
dynamic spatial masks. Dynamic masks follow an A → B → A schedule across the
four one-second windows, with exactly the declared number of channels in each
window. The default uses five mask repeats per degraded scenario and one full
scenario: **61 inference conditions per partition per model**, after training.
This increases inference work, not the number of training runs.

Preprocessing uses four-second cue-aligned trials, artifact exclusion, 2–30 Hz
zero-phase filtering independently inside each availability window, and
train-only channel normalization. Window-local filtering prevents hidden
intervals from influencing observed intervals. Models receive the **entire
four-second trial** after masking/completion. This is an offline within-session
study. It differs from the competition's train-on-T/test-on-E benchmark.

## Presets

| Configuration | Models | Participants | Seeds | Training runs | Output |
|---|---:|---:|---:|---:|---|
| [Full](configs/study.json) | 4 | 9 | 3 | **108** | `results/inm-v3` |
| [Pilot](configs/pilot.json) | 4 | 9 | 1 | **36** | `results/inm-v3-pilot` |
| [Debug](configs/debug.json) | EEGNet only | 3 | 1 | **3** | `results/inm-v3-debug` |
| [Smoke](configs/smoke.json) | 4 | 1 | 1 | **4** | `results/inm-v3-smoke` |

Debug and smoke limit training to 12 epochs and masks to one repeat; smoke
exercises all four model paths on one participant. Their scores are for workflow
debugging. Presets inherit `study.json` through
`extends`. For another small run, select any subset of the four model keys,
participants and seeds in a preset, with a **new output directory**. Backbone
options are shared by each baseline/tensor pair. Attention cannot be switched.

## Setup

Requires Python **3.12+**, CUDA-enabled PyTorch and the packages in
[requirements.txt](requirements.txt). Training requires an available CUDA GPU.
Optional plots require Matplotlib 3.8 or newer, below version 4.

```bash
git clone https://github.com/georgiipr/agfl-inm.git AGFL-inm
cd AGFL-inm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

An existing environment with these dependencies can also be used. Match the
PyTorch build to the available GPU drivers.

## Dataset setup: BCI Competition IV, data set 2a

The dataset has nine participants, four motor-imagery classes (left hand, right
hand, feet and tongue), and 22 EEG channels sampled at 250 Hz. The three EOG
channels are excluded from classifier inputs. See the
[official dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf).

1. Open the [official download page](https://www.bbci.de/competition/iv/#download),
   follow **Data sets 2a: GDF files zipped**, and observe its usage/citation terms.
2. Extract `A01T.gdf` through `A09T.gdf` with their original names into `ml/`
   beside the project. These T-session recordings contain the cue labels; this
   study needs no E-session recordings or external evaluation-label files.
3. Set `data.data_dir` in `configs/study.json` if using another directory.
   Relative data/output paths resolve from the project launch directory,
   including for inherited presets. Keep recordings out of version control.

```text
workspace/
├── AGFL-inm/
│   ├── run.py
│   ├── configs/
│   └── results/              # generated; ignored by Git
└── ml/
    ├── A01T.gdf
    ├── ...
    └── A09T.gdf
```

## Run experiments

Run commands from the repository root. Inspect a preset and run one
participant/seed task:

```bash
python run.py --config configs/smoke.json --plan
python run.py --config configs/smoke.json --task-index 0
```

Each task trains every configured model for one participant and seed, then
evaluates the selected checkpoints under all configured channel-loss scenarios.
To run the full preset sequentially in Bash:

```bash
config=configs/study.json
task_count=$(python run.py --config "$config" --task-count)
failed=0
for ((task=0; task<task_count; task++)); do
    python run.py --config "$config" --task-index "$task" || failed=1
done
test "$failed" -eq 0
```

Use `configs/debug.json` or `configs/pilot.json` for smaller runs. Completed
runs and selected checkpoints are verified before reuse; interrupted evaluation
resumes from the saved selected checkpoint. Changed source, environment or
settings require a new output directory.

## Results and plots

Reports update after model runs. Read `results/inm-v3/report/summary.md` and
`progress.json`. `accuracy_by_run.csv` shows each completed participant/seed;
`accuracy_by_subject.csv` and `accuracy_table.csv` show complete participant and
overall averages. `paired_tensor_gain.csv` compares each tensor model with its
matching backbone. Validation and test results stay separate.

Rebuild reports and optional PNG/PDF plots from saved records:

```bash
python run.py --config configs/study.json --summarize-only --plots
```

Use the same preset for a debug or pilot study. Figures appear in
`report/plots/`: validation/test channel-loss curves for the configured models
and paired tensor robustness gains. `plots_status.json` records skipped
incomplete comparisons. Plotting reads saved tables and never retrains models.

Each `artifacts/Axx_seed_s/` contains splits, dataset metadata, normalization,
Tucker factors and factor history. Each `MODELS/<model>/` contains
`checkpoint.pt`, its checksum/selection record, `history.json`, configuration
and all scenario metrics. Keep these with `study.json` when archiving results.

## Code map

- `eeg_models/models/`: four concrete models; fixed MHA lives inside their backbones.
- `eeg_models/models/_shared/input.py`: explicit masks and tensor completion before encoding.
- `eeg_models/models/_shared/tensor.py`: training-only Tucker factors and masked inference.
- `inm/study.py`: participant/seed workflow, checkpoints and evaluation sweep.
- `inm/training.py`: end-to-end fitting and full-channel validation selection.
- `inm/availability.py`: reproducible static/dynamic channel-loss masks.
- `inm/reporting.py`, `inm/plots.py`: completeness-aware tables and optional figures.
- [Experiment protocol](docs/experiment.md), [model structure](docs/model_adaptation.md),
  [tensor mathematics](docs/tensor_math.md), and [availability masks](docs/availability_protocol.md).
