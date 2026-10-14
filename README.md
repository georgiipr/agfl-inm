# AGFL-inm

EEG classification with **EEGNet** and **Signal Transformer**, each evaluated
alone, with **frozen covariance completion**, and with **learned covariance
completion**. Four-head **MHA is built into both backbones**. Covariance replaces
Tucker in the active experiment; historical Tucker results remain below.

The separate **ATCNet published-checkpoint validation** is now available in
[`published_covariance/`](published_covariance/). It compares zero filling,
frozen covariance and learned covariance using the research branch's official
pretrained ATCNet checkpoints and T-to-E protocol. Read the
[validation goal, protocol, dataset setup and launch instructions](docs/published-covariance/README.md).
It uses an existing TensorFlow GPU environment. Its results are reported
separately from the six-model PyTorch study below; no reproduced accuracy is
claimed before execution.

Two independent ATCNet student studies answer the reduced-channel training
questions: **Q1** trains directly on incomplete 16/6-channel inputs using class
labels; **Q2** trains the same students with additional guidance from the frozen
22-channel teacher. They have separate configurations, outputs and experiment
selectors. See [student goals, protocol and commands](docs/channel-students.md).
The covariance study remains independently runnable.

## Goals

1. Preserve the restored full 22-channel classification baseline.
2. Compare zero filling, training-fitted covariance and classification-informed
   covariance under identical trials, splits, preprocessing and outage masks.
3. Measure degradation at **22 → 16 → 11 → 6 channels**, including static
   random/spatial loss and dynamic disappearance/return of electrodes.
4. Train participants individually and report all nine with equal weight,
   including unfavorable outcomes. Reconstruction quality alone does not
   establish classification benefit.

## Six model variants

| Model key | Implementation |
|---|---|
| `eegnet` | Restored whole-trial EEGNet with electrode MHA |
| `eegnet_covariance_frozen` | The same selected EEGNet with train-only covariance completion |
| `eegnet_covariance_learned` | The same selected EEGNet with supervised train-only covariance adaptation |
| `signal_transformer` | Per-electrode temporal tokenizer, spatial MHA blocks and learned spatial readout |
| `signal_transformer_covariance_frozen` | The same selected Signal Transformer with frozen covariance |
| `signal_transformer_covariance_learned` | The same selected Signal Transformer with learned covariance |

Backbone architecture, MHA and temporal processing remain unchanged. Completion
fills only missing samples before encoding, preserves measured samples exactly,
and retains the original availability mask for MHA keys. With all channels
present it is bypassed exactly. The missing-channel input still has the original
22 electrode positions and entire four-second trial.

## Experiment pipeline

For **each participant and seed**:

1. Load that participant's BCI IV 2a T-session trials. Keep the original
   stratified 60/20/20 training/validation/test split and artifact inclusion.
2. Fit channel normalization and a shared 22×22 channel second moment using
   **training trials only**. Frozen and learned covariance start from that same
   matrix and use the same conditional-completion solver.
3. Train each selected **backbone once on all 22 channels**, end to end. Restore
   **500 epochs, batch size 64, AdamW learning rate 0.001 and weight decay 0.001**,
   balanced class weights, ten warmup epochs/cosine decay, original initialization
   and shuffled batch order, no early stopping and no gradient clipping.
   **Lowest full-channel validation loss selects the backbone checkpoint**;
   exact ties retain the earlier epoch. All three variants reuse those weights.
4. Frozen covariance requires no supervised fitting. For learned covariance,
   freeze the selected backbone, including BatchNorm and dropout, and optimize
   only its **253 Cholesky coordinates** for **100 fixed epochs**. Simulated
   training outages sample 22/16/11/6 retained channels and all four outage
   patterns. Training subset banks are disjoint from validation/test banks.
   This phase uses Adam, batch size 32, learning rate 0.001, no weight decay,
   clipping at 1, and classification loss plus 0.1 hidden-training MSE plus
   0.0001 displacement from initialization. Training class weights are retained.
5. Retain the **final covariance calibration epoch**, without early stopping
   or degraded-validation selection. Then freeze completion and evaluate all
   variants on validation and held-out test at 22/16/11/6 channels. No fitting
   occurs during either evaluation sweep.
6. Save checkpoints, backbone reuse identity, calibration state, training
   histories, completion masks/order, input/mask hashes and source provenance.
   Average repeats, then seeds within participant, then participants equally.
   Reports compare both covariance variants against the baseline and learned
   against frozen covariance. Incomplete means remain blank.

The extra train-only adaptation in the learned arm is an intentional training
intervention. Its **fixed final-epoch rule differs from the research branch's
selection on degraded validation**, preserving this setup's descriptive-only
use of degraded validation. The learned arm does not test robustness to outage
families unseen during calibration; its evaluation uses unseen trials and
partition-reserved electrode subsets. Frozen covariance retains the full-only
training regime. Test scores never select weights or change the protocol.

The existing evaluation still has five repeats per degraded scenario: static
random, static spatial, dynamic random and dynamic spatial at 16/11/6 channels,
plus one full-input condition. Dynamic masks follow A → B → A across four
one-second windows. This is **61 inference conditions per partition per variant**,
without retraining at each channel count.

Full-input preprocessing keeps the original **2–30 Hz zero-phase run-level
bandpass before trial extraction**. Degraded inputs remove hidden raw intervals
before filtering each observed electrode span; no filter crosses an outage,
native gap or run boundary. Outside-trial recording context remains observed.
Normalization is train-only, fitted in float64 and applied in float32.
Execution retains one CPU thread and deterministic CUDA settings.
This is an offline within-session study, not causal streaming or the
competition's train-on-T/test-on-E benchmark.

## Presets

| Configuration | Variants | Participants | Seeds | Backbone fits | Learned covariance fits | Output |
|---|---:|---:|---:|---:|---:|---|
| [Full](configs/study.json) | 6 | 9 | 3 | 54 | 54 | `results/inm-covariance-v6` |
| [Pilot](configs/pilot.json) | 6 | 9 | 1 | 18 | 18 | `results/inm-covariance-v6-pilot` |
| [Debug](configs/debug.json) | EEGNet baseline | 3 | 1 | 3 | 0 | `results/inm-covariance-v6-debug` |
| [Smoke](configs/smoke.json) | 6 | 1 | 1 | 2 | 2 | `results/inm-covariance-v6-smoke` |
| [EEGNet: A03/A04/A09](configs/eegnet_3subjects.json) | 3 | 3 | 1 | 3 | 3 | `results/eegnet-a03-a04-a09-covariance-v6` |
| [EEGNet: all participants](configs/eegnet_all_subjects.json) | 3 | 9 | 1 | 9 | 9 | `results/eegnet-all9-covariance-v6` |

The next EEGNet experiments use **seed 0**, with **9 model evaluations** for
A03/A04/A09 and **27** for all nine. They produce two independent reports;
the all-nine experiment includes A03/A04/A09 again. Signal Transformer is
retained but excluded from these two presets.

Debug/smoke use 12 backbone epochs, three covariance calibration epochs and
one mask repeat. Their scores are only for workflow debugging. Presets inherit
through `extends`. Select smaller model/participant/seed subsets with a new
output directory. A covariance-only preset still trains its required backbone.

## Historical reference results: Tucker (v5)

These are preserved measurements from the previous Tucker experiment, not covariance results.
The tables in this section describe historical Tucker runs. Covariance runs
use their own output directories and manifests.

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

The original reports remain in `results/eegnet-a03-a04-a09-restored-v5`
and `results/eegnet-all9-restored-v5`. Those directories are not reused by the
covariance presets. Preserve their recorded configuration/source manifest when
archiving them; the current presets describe the new experiment.

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

Run from the repository root in the existing GPU environment:

```bash
python run.py --config configs/eegnet_3subjects.json --plan
python run.py --config configs/eegnet_3subjects.json --task-index 0
```

Each task owns one participant/seed and evaluates every configured variant.
To execute a preset sequentially in Bash:

```bash
config=configs/eegnet_3subjects.json
task_count=$(python run.py --config "$config" --task-count)
failed=0
for ((task=0; task<task_count; task++)); do
    python run.py --config "$config" --task-index "$task" || failed=1
done
test "$failed" -eq 0
```

Use `configs/eegnet_all_subjects.json` for the independent all-nine experiment.
Use `configs/smoke.json` to exercise all six paths with the short budget.
The supplied launcher first runs the GPU covariance preflight in
`tests/test_covariance.py`; it aborts on failure before expensive training.
Interrupted evaluation restores completed selected checkpoints. Completed
backbones are reused within their own study; interrupted covariance calibration
restarts deterministically from initialization. Changed source/environment or
settings require a fresh output directory.

## Results and plots

Each preset writes its own `report/summary.md` and `progress.json`.
`accuracy_by_run.csv`, `accuracy_by_subject.csv`, `accuracy_table.csv` and
`robustness_table.csv` retain individual runs, participant means, channel-loss
curves and robustness. `paired_completion_gain.csv` contains frozen/learned
covariance-minus-baseline and learned-minus-frozen gains with participant
bootstrap intervals. Validation and test stay separate; unfinished means remain
blank. Each report is tied to its recorded source, configuration and runtime.

Rebuild reports and optional PNG/PDF plots from saved records:

```bash
python run.py --config configs/eegnet_3subjects.json --summarize-only --plots
python run.py --config configs/eegnet_all_subjects.json --summarize-only --plots
```

Figures appear in each `report/plots/`: validation/test availability curves and
paired covariance gains. `plots_status.json` identifies incomplete comparisons.
Plotting reads saved tables and never retrains models.

Each `artifacts/Axx_seed_s/` contains the split, dataset metadata, normalization
and covariance calibration. `BACKBONES/<backbone>/` stores the shared selected
checkpoint/history; `MODELS/<model>/` stores the complete variant checkpoint,
selection record and metrics. Learned variants additionally retain
`completion_history.json`, `completion_fit.json` and `completion_training.npz`.
Keep these with the study manifest when archiving results. Generated artifacts
and recordings are excluded from Git.

## Code map

- `eeg_models/models/`: two backbones and their frozen/learned covariance variants.
- `eeg_models/models/_shared/input.py`: explicit masks and completion before encoding.
- `eeg_models/models/_shared/covariance.py`: training-only moment, SPD covariance and conditional solve.
- `inm/study.py`: shared backbone checkpoints, calibration and evaluation sweeps.
- `inm/training.py`: restored end-to-end backbone recipe and full-input validation selection.
- `inm/completion_training.py`: fixed-budget supervised covariance calibration.
- `inm/availability.py`, `inm/data.py`: reproducible masks and outage-aware preprocessing.
- `inm/reporting.py`, `inm/plots.py`: complete-cohort comparisons and optional figures.
- `published_covariance/`: isolated ATCNet checkpoint reproduction, cluster admission
  checks, covariance calibration, audited T-to-E scoring and independent verification.
- [Experiment protocol](docs/experiment.md), [model structure](docs/model_adaptation.md),
  [covariance mathematics](docs/covariance_math.md), [availability masks](docs/availability_protocol.md).
