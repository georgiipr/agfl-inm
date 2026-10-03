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
   by lowest validation loss using the training class weights. Exact ties keep
   the earlier epoch; balanced accuracy remains the primary reported metric.
4. Freeze the selected checkpoint. Evaluate **22 → 16 → 11 → 6 channels** on
   validation, then held-out test, without further fitting or retraining.
5. Record all scenarios and repeats, selected weights, training history, split,
   preprocessing/factor state, source hashes and configuration provenance.
6. Average mask repeats, then seeds within each participant, then give each
   configured participant equal weight. Report participant variation and paired
   tensor-minus-baseline gains. Incomplete averages remain blank.

Degraded validation is descriptive after selection. It does not select epochs,
ranks or settings; held-out test scores never influence those decisions.
Each decreased count has static random, static spatial, dynamic random and
dynamic spatial masks. Dynamic masks follow an A → B → A schedule across the
four one-second windows, with exactly the declared number of channels in each
window. The default uses five mask repeats per degraded scenario and one full
scenario: **61 inference conditions per partition per model**, after training.
This increases inference work, not the number of training runs.

Preprocessing uses four-second cue-aligned trials, includes marked artifact
trials, and restores the original 2–30 Hz zero-phase bandpass **over each
recording run before cutting trials**. Full-channel training, validation and
test use those original inputs. Channel normalization uses training trials only.
The one-second availability windows define outages and tensor axes; they do
not divide full-channel filtering or model encoding.

Degraded inputs are built from the raw recording: remove missing intervals
before filtering each observed electrode span. Filtering never crosses an
outage, native recording gap or run boundary. Uninterrupted electrodes reuse
their original filtered values. Outages apply inside the cue-aligned trial;
recording context outside it remains observed. The same preprocessed inputs
and masks reach every model, with input hashes recorded for pairing checks.
Models receive the **entire four-second trial** after masking/completion.
This is an offline within-session study with available recording context,
not causal streaming or the competition's train-on-T/test-on-E benchmark.

The production recipe uses **500 epochs without early stopping**, batch size
**64**, AdamW with learning rate **0.001** and weight decay **0.001**, balanced
training-class weights, ten warmup epochs followed by cosine decay, and no
gradient clipping. Batch order uses the original seeded PyTorch DataLoader
shuffling. Both backbones use fixed four-head MHA with learned Q/K/V and output
linear projections initialized in independent per-layer RNG streams,
preserving the surrounding backbone's initialization. Normalization statistics
are fitted in float64 on training trials and applied in float32. Execution uses
one CPU thread and the original deterministic CUDA settings; runtime versions
and GPU names are recorded.

The EEGNet full-channel route retains the original pre-spatial architecture,
initialization, preprocessing and training recipe. Tensor completion and key
masking are bypassed for full inputs. Reproducing a historical score also needs
its participant set, seeds and runtime environment. The recorded v5 reference
comparison is reported below. Distinct output directories preserve each study's
configuration and provenance.

## Latest results (v5)

Two completed experiments compare **EEGNet with built-in MHA**, with and
without tensor completion: A03/A04/A09 (**6/6 model runs**) and all nine
participants (**18/18 model runs**). Both use **seed 0**, the 500-epoch recipe
above, and five mask repeats per degraded scenario. Signal Transformer is not
included in these results.

These are **held-out T-session test results**, using individually trained
participants and stratified splits of 176 training, 56 validation and 56 test
trials. Each test partition contains 14 trials per class, so balanced accuracy
and ordinary accuracy are equal. Full-channel validation loss selects the
checkpoint; neither degraded validation nor test scores select it.

### Full-channel performance

Both EEGNet variants have identical full-channel training histories and test
scores, since completion is bypassed when all 22 channels are available.

| Participant | EEGNet / EEGNet + tensors test accuracy (%) | Selected epoch |
|---|---:|---:|
| A01 | 67.86 | 135 |
| A02 | 57.14 | 188 |
| A03 | 67.86 | 281 |
| A04 | 39.29 | 25 |
| A05 | 30.36 | 81 |
| A06 | 33.93 | 61 |
| A07 | 69.64 | 120 |
| A08 | 67.86 | 136 |
| A09 | 71.43 | 268 |

The A03/A04/A09 mean is **59.52%** (participant SD **17.62 pp**); the all-nine
mean is **56.15%** (SD **16.85 pp**). For the matched A03/A04/A09 seed-0
reference, dataset fingerprints, split IDs, selected epochs, full-channel
metrics and common training-history metrics across all 500 epochs reproduce
the archived original EEGNet/MHA run exactly. This comparison is specific to
that cohort and seed.

### Accuracy as channel availability decreases

For each degraded channel count, average the five repeats within each outage
pattern, then the four patterns (static random, static spatial, dynamic random,
dynamic spatial), then give each participant equal weight. Gains are tensor
minus baseline in **percentage points (pp)**. Each model trains on 22 channels
once; the same selected checkpoint serves the entire availability sweep.

| Participants | Channels retained | EEGNet (%) | EEGNet + tensors (%) | Tensor gain (pp) |
|---|---:|---:|---:|---:|
| A03/A04/A09 | 22 | 59.52 | 59.52 | 0.00 |
| A03/A04/A09 | 16 | 40.86 | 45.77 | +4.91 |
| A03/A04/A09 | 11 | 36.01 | 38.81 | +2.80 |
| A03/A04/A09 | 6 | 35.77 | 35.68 | -0.09 |
| All nine | 22 | 56.15 | 56.15 | 0.00 |
| All nine | 16 | 38.47 | 41.57 | +3.10 |
| All nine | 11 | 34.53 | 36.23 | +1.70 |
| All nine | 6 | 32.18 | 32.12 | -0.06 |

Tensor completion provides a modest average benefit at 16 and 11 retained
channels. At six channels, its average benefit is effectively absent in both
cohorts.

### Paired robustness gains: all nine participants

Robustness equally weights accuracy at **22, 16, 11 and 6 channels** within
each outage pattern. Positive differences favor tensor completion. Pointwise
95% percentile intervals use 2,000 participant-bootstrap resamples; participants,
not mask repeats, are the resampling units.

| Outage pattern | Tensor gain (pp) | 95% bootstrap interval (pp) |
|---|---:|---|
| Static random | +0.84 | [-0.07, 1.89] |
| Static spatial | +0.71 | [-0.00025, 1.45] |
| Dynamic random | +1.99 | [0.95, 3.33] |
| Dynamic spatial | +1.18 | [0.51, 1.90] |

Dynamic outages show the clearest benefit in this experiment. Both static
robustness intervals include zero. The intervals are exploratory and have no
multiple-comparison correction; **one training seed** cannot establish
stability across initialization seeds. These results are offline, within-session
availability measurements, not competition T-to-E or unseen-participant results.

The values above come from `accuracy_by_subject.csv`, `accuracy_table.csv` and
`paired_tensor_gain.csv` in the generated reports for
[the three-participant preset](configs/eegnet_3subjects.json) and
[the all-nine preset](configs/eegnet_all_subjects.json). Detailed records and
plots remain in their respective generated result directories, which are
excluded from version control.

## Presets

| Configuration | Models | Participants | Seeds | Training runs | Output |
|---|---:|---:|---:|---:|---|
| [Full](configs/study.json) | 4 | 9 | 3 | **108** | `results/inm-v5` |
| [Pilot](configs/pilot.json) | 4 | 9 | 1 | **36** | `results/inm-v5-pilot` |
| [Debug](configs/debug.json) | EEGNet only | 3 | 1 | **3** | `results/inm-v5-debug` |
| [Smoke](configs/smoke.json) | 4 | 1 | 1 | **4** | `results/inm-v5-smoke` |
| [EEGNet: A03/A04/A09](configs/eegnet_3subjects.json) | EEGNet and its tensor variant | 3 | 1 | **6** | `results/eegnet-a03-a04-a09-restored-v5` |
| [EEGNet: all participants](configs/eegnet_all_subjects.json) | EEGNet and its tensor variant | 9 | 1 | **18** | `results/eegnet-all9-restored-v5` |

The two EEGNet comparison presets use seed 0 and the restored **500-epoch**
production recipe, with five mask repeats for each degraded condition. They
produce independent reports in separate output directories. The all-participant
preset trains all nine participants, including A03/A04/A09 again.

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

Reports update after model runs. Read `results/inm-v5/report/summary.md` and
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
- `inm/data.py`: original run-filtered trials and cached raw-mask-aware outage inputs.
- `inm/reporting.py`, `inm/plots.py`: completeness-aware tables and optional figures.
- [Experiment protocol](docs/experiment.md), [model structure](docs/model_adaptation.md),
  [tensor mathematics](docs/tensor_math.md), and [availability masks](docs/availability_protocol.md).
