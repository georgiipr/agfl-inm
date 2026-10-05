# Encoder candidate study contract

## Purpose and scope

Test three proposed classification designs against fresh local and spatial
controls. This is a new supervised study in `inm/encoder_candidates/`, not an
extension or replay of the completed tensor study. The ideas are hypotheses;
neither stronger accuracy nor a Tucker gain is assumed. Do not implement Tucker,
reconstruction losses, rank searches, cross-subject pretraining, or T/E transfer.

Sessions 01–09 implement and test software using tiny synthetic data. A human
then runs and inspects the real pilot and cohort. Session 10 reviews complete
real evidence. Implementation completion does not mean experiments have run.
The default automation stops after session 09.

The user authorizes end-to-end training and spatial/temporal mixing in this new
package after unavailable raw samples are selected out. These exceptions never
change the legacy invariants. The two local encoders must preserve channel and
window locality at frozen feature extraction; their head may mix electrodes.
Spatial model activations are latent features, not original electrode features.
Never mask a full-input feature cache to simulate raw loss after spatial mixing.

## Fixed experiment and comparisons

Use nine participants and seeds [0,1,2], subject-major order: 27 tasks. Each task
fits exactly these five full-input neural arms, in this order:

| Arm ID | Construction | Interpretation |
|---|---|---|
| `local_control` | Existing EEGWindowEncoder plus existing MHA FeatureClassifier | Fresh local reference |
| `local_power` | New multiscale log-variance local encoder plus the SAME head | Primary local representation comparison |
| `spatial_eegnet` | Current EEGNetClassifier with flatten head and mask conditioning | Fresh spatial reference |
| `spatial_filterbank` | Fixed filter bank, learned spatial filters, log-variance, linear head | FBCNet-inspired candidate |
| `spatial_transformer` | Same EEGNet front-end as spatial reference plus small Transformer head | Exploratory complete-head comparison |

All five are trained from scratch: 135 neural fits. For each selected local arm,
fit one ordered-feature logistic probe and one shuffled-training-label control:
4 CPU fits/task, 108 CPU fits total. No automatic repetitions or hyperparameter
search. All arms have full-input training. Mixed-availability training is a
separate future study, not an extra dimension silently added here.

Primary contrast: local_power minus local_control. Secondary contrasts:
spatial_filterbank minus spatial_eegnet and spatial_transformer minus
spatial_eegnet. Compare clean full-input validation balanced accuracy (BA), then
log-loss descriptively. Predeclare a candidate as promising for independent
confirmation if its complete paired cohort mean improves by at least 2 percentage
points and at least 6/9 participant differences are positive. This is a proposed
practical screening rule, not a significance test or a definition of strong BCI
performance. Report every contrast and negative result. Use 2000 participant
bootstrap resamples, RNG seed 20261003, for exploratory 95% intervals; disclose
multiple comparisons and reused validation. No tuning from test outcomes.

Capacity, pooling, regularization, and optimizability differ. The local comparison
changes an encoder design, not just one causal factor. The Transformer comparison
changes the whole readout. Historical 60.86% validation BA is context only; fresh
paired controls are the denominators. Paper accuracies use other protocols.

## Data and historical inputs

Use T-session recordings only, 22 electrodes in canonical order, 250 Hz, cue
offset zero, artifact exclusions, four non-overlapping 250-sample windows, and
the existing window-local 2–30 Hz preprocessing. No augmentation in this study.
Reuse the persisted participant/seed split membership from
`results/baselines-reproducible-v1/artifacts/Axx_seed_s/split.json`; do not call
an original study writer, regenerate membership, or write into input directories.
Read its `dataset.json` and study manifest to verify recording checksums, stable
IDs, labels, exclusions, channel order, and declared preprocessing. Copy original
metadata hashes into the new provenance. Conflicting data or IDs block real fits.

The old checkpoint source mismatch is NOT a prerequisite for this new training
study: old checkpoints and histories are not inputs. Use current, pinned source
for new fits and record its own identity. Do not claim historical reproduction
or edit historical manifests to match it. If current loading cannot match the
recorded dataset identity, stop and diagnose it; do not silently re-split.

Load recordings as needed, but immediately select train and validation into the
public PreparedData object. Retain held-out membership only for overlap checks;
never expose, normalize, score, summarize, or use test arrays/labels downstream.
Fit per-channel mean and population std using training trials/time alone, floor
std at 1e-8, and apply to validation. No per-trial or per-window power scaling.
Persist raw recording hashes, original metadata hashes, exact partition IDs,
training normalization, filtered dataset fingerprint, and split identity.

All arms get identical ordered examples, batches, and evaluation masks. Seed
model initialization independently of data-order RNG; resetting each arm must
not let its parameter count change the minibatch stream. Initialize the shared
local heads identically. Initialize shared EEGNet front-end tensors identically
for the spatial reference/Transformer pair, then train them independently.

## Exact model proposals

All model forward methods accept normalized raw `[B,22,4,250]` and Boolean
mask `[B,22,4]` or None. Select using torch.where BEFORE all arithmetic or
filtering. Hidden NaNs must not leak. Reject nonfinite observed input and an
all-missing window. Preserve electrode identity and return logits `[B,4]`.

Local control wraps `inm.model.EEGWindowEncoder` with its current defaults:
f1=16, d=2, f2=32, kernel_length=32, pools 8/16, dropout=0.5. Reshape to its
existing `[B,22,1000]` interface. Both local models use FeatureClassifier with
representation=baseline, MHA, dim=32, heads=4, depth=1, dropout=0.1 and its
existing spatial/window pooling. Disclose its missing-token attention behavior.
Do not add an ordered/temporal head to only the new local arm.

Local power: four shared Conv1d branches, one input channel and eight filters
each, kernels [16,32,64,128], stride 1, explicit zero same-padding, no bias.
Operate on `[B*22*4,1,250]`. For each filter compute population variance over
the 250 responses and `log(variance + 1e-6)`, concatenate to 32 features and
reshape `[B,22,4,32]`. No rectification, temporal average pooling, or input
instance normalization before variance. Zero hidden output features explicitly.
Store epsilon and layout in constructor settings. This is our proposed local
power model, not FBCNet. The first experiment preserves both feature count nor
head; additional normalization/learned pooling would be another experiment.

Spatial reference wraps current `inm.baselines.eegnet.EEGNetClassifier`:
f1=8, depth_multiplier=2, f2=16, temporal_kernel=64, separable_kernel=16,
pooling [4,8], dropout=0.5, head_type=flatten, mask_conditioned=True; other
constructor defaults unchanged. It may mix temporal windows after raw masking.

Spatial filter bank: six fixed bands [[4,8],[8,12],[12,16],[16,20],[20,24],
[24,30]] Hz within the common input passband. Design 81-tap Hamming FIR filters
with scipy.signal.firwin(pass_zero=False, fs=250), retain coefficients as frozen
buffers, apply same-padded convolution independently per channel/window. Then
learn four spatial filters per band spanning 22 electrodes (24 outputs/window),
max-norm 1 per spatial filter. Compute log(population variance + 1e-6) over each
250-sample response, flatten ordered windows (96 features), dropout=0.5,
linear four-class head. This simplified window-local FBCNet-inspired adaptation
is not a paper reproduction. Do not filter across a missing-window boundary or
expand the preprocessing band beyond 30 Hz. Include availability flags at the
classifier input (88 flags), with the same raw masking convention as reference.

Spatial Transformer: reuse the spatial reference's `forward_features` to obtain
31 temporal positions with 16 features. Retain only its front-end modules, not
an unused original classifier; report parameters actually used. Project to d_model=32, add fixed
sinusoidal temporal positions, one TransformerEncoderLayer with 4 heads,
dim_feedforward=64, GELU, dropout=0.1, norm_first=True, batch_first=True.
Apply final LayerNorm, mean over positions, concatenate the 88 availability
flags, and use a linear four-class head. Train the entire front-end. Do not add
CLS tokens, deeper layers, learned frequency banks, or a sweep. Describe it as
a compact EEGNet–Transformer adaptation, not an EEG Conformer reproduction.

## Training and diagnostics

Loss: ordinary unweighted four-class cross-entropy. AdamW, lr=0.001,
weight_decay=0.0001, batch_size=32, maximum 250 epochs, minimum 75 epochs,
patience 50 epochs without improvement, gradient global norm clip=1. Warm up
linearly for 10 epochs then cosine-decay to zero by epoch 250; record exact
per-epoch LR. Compare BA without rounding; tie-break on lower validation
log-loss, then earlier epoch. Epoch selection uses full-input validation only.
Save full selected models, constructors, buffers, histories, class/channel order,
normalization and provenance. Store numeric arrays and primitive checkpoint
metadata so weights_only=True reload is supported. No original model changes.

Record clean eval-mode train and validation BA, accuracy, macro-F1, log-loss,
per-class recall/confusion matrix at selected checkpoints, and separately the
optimization-time loss/accuracy. Verify prediction equality after checkpoint
reload (rtol=1e-4, atol=1e-5). Report parameters, best epoch, time and failures.
All finite gradients must pass before stepping; call existing model clipping
when applicable. Synthetic fixtures may use tiny training budgets only when
explicitly synthetic; real configs must not accept silent budget overrides.

For each selected local encoder, freeze parameters, BatchNorm and dropout and
extract full-input `[N,22,4,32]`. Fit StandardScaler on training features and L2
logistic regression C=1, solver=lbfgs, max_iter=1000, tol=1e-4, random_state=0,
no class weights. Flatten ordered features. The second probe permutes training
labels with seed+700001; evaluate both against original labels. Save scaler,
coefficients, classes, IDs and probabilities as numeric NPZ with allow_pickle=False.
Nonconvergence is an incomplete diagnostic, not success. A single shuffle is
not a p-value and need not produce exactly 25% BA. Probe failure cannot prove
absence of nonlinear information. Save the encoder AND head for future audits.

Evaluate selected neural models on validation only under full_22 once plus
random_static_16, dynamic_random_16, random_static_6, dynamic_random_6 with five
repeats each. Use existing make_mask_bank with the same subject/seed/partition/
stable IDs for every arm. Re-run every model from masked raw input; never use
full-input spatial caches. The dynamic schedule is the existing A-B-A-A rule.
These are secondary robustness diagnostics, not epoch/architecture selectors.
No 11-channel or spatial-loss validation conditions. No test outputs.

## Stable interfaces and files

Keep plan/preflight imports stdlib-only. Create only the package, config,
documentation and tests named by the sessions; do not modify old study code.

| Module | Interface |
|---|---|
| protocol | load_config(path), tasks(cfg), arms(cfg), study_identity(cfg) |
| data | prepare_subject(cfg, subject, seed, directory, loader=None) -> PreparedData |
| models | build_model(arm_id, seed), restore_model(arm_id, constructor, state_dict) |
| local | LocalClassifier(kind), LocalPowerEncoder with forward_features/freeze |
| spatial | SpatialEEGNet, SpatialFilterBank |
| transformer | SpatialTransformer |
| training | fit_classifier(model, prepared, cfg, arm_id, directory, device) -> result |
| diagnostics | evaluate_selected(model, prepared, cfg), fit_local_probes(model, prepared, cfg, directory) |
| study | run_task(cfg, task_index, device), run_smoke(cfg, output_dir) |
| reporting | summarize(cfg) -> complete/partial report paths |

PreparedData has subject/seed, train/validation partition objects (raw normalized
signals, labels, sample_ids), normalization, split_id, data_id, provenance. No
test partition field. Models provide constructor_settings(), forward_features()
and clip_weights() (explicit no-op where none), plus forward(raw, mask=None).
models.py uses lazy imports and is the sole five-arm factory, never eval of names.

Create `configs/encoder-candidates.json`, schema_name `agfl-encoder-candidates-v1`,
name `encoder_candidates_full_v1`, subjects 1–9, seeds [0,1,2], partitions
[train,validation], training_regime full. Include all fixed model/training/probe/
condition/selection settings in this contract; unknown settings fail. Paths are
relative to config: data_dir ../../ml, split_source_dir
../results/baselines-reproducible-v1, output_dir ../results/encoder-candidates-v1.
Reject output overlap both directions with any input, the repository root,
existing incompatible outputs, and symlink aliases. Plan prints 27 tasks,
135 neural fits, 108 CPU probe fits, zero Tucker fits. Preflight reads input
metadata/paths, never imports numerical packages or deserializes checkpoints.

CLI: `python -m inm.encoder_candidates --config CONFIG` with mutually exclusive
--plan, --preflight, --task-index N, --smoke, --summarize-only; --device defaults
to cpu, cuda explicit; --output-dir is smoke-only. Missing future commands fail
explicitly until implemented. Real fits require an explicit task command outside
coding sessions. Smoke uses a fresh temporary output and synthetic:true, runs
all five models through tiny training/reload/diagnostics, and cannot satisfy the
real evidence gate. Cap CPU threads in synthetic tests for predictable runtime.

## Artifact and evidence contract

Atomic writes; preserve failures; reject partial/corrupt/incompatible reuse.
Freeze source/config/packages across real execution. study_id is SHA-256 of
canonical JSON of configuration, source map and package versions plus verified
input identities. Record the exact root *.py and all agfl/inm *.py SHA-256 map.
This new source identity does not alter any old result or receipt.

The session-10 gate reads `results/encoder-candidates-v1/report/evidence.json`:
schema_name `agfl-encoder-candidates-report-v1`, status complete, synthetic false,
partitions [train,validation], training_regime full, subjects 1–9, seeds [0,1,2],
arms equal the ordered five IDs above, study_id (64 lowercase hex), config_sha256
(raw config bytes), packages (nonempty version-string map), source_files_sha256,
and tasks (27 records {subject,seed,path,sha256}, relative to output root).

Each task is `tasks/Axx_seed_s/task.json`, schema_name
`agfl-encoder-candidates-task-v1`, same status/synthetic/partitions/training_regime/
study_id/config_sha256/source_files_sha256 as report, subject/seed, split_id and
data_id (64 lowercase hex), and checks with EXACT Boolean keys:
`split_isolation`, `normalization_train_only`, `raw_masking`, `checkpoint_reload`,
`probe_reload`, `mask_pairing`, `identities`. True means executed verification,
not assumed readiness. Fits is five records {arm,path,sha256}, paths relative to
task directory. Task metadata also retains original input metadata and recording
checksums. No empty or synthetic task may enter a real complete report.

Each fit is `ARMS/<arm>/result.json`: status complete, synthetic false,
partitions [train,validation], subject/seed/arm, study_id/config_sha256/split_id/
data_id equal task, selected_epoch, parameter_count, clean_train metrics, and
validation list of 21 rows with scenario, repeat, mask_sha256, balanced_accuracy,
log_loss plus other metrics. Full_22 repeat=0, each degraded scenario repeats
0–4. All arms within task share exact mask hashes. artifact_sha256 maps paths
relative to the fit directory to file hashes, including checkpoint.pt,
history.json, predictions.npz. Local arms additionally include probe.npz and
probe_shuffled.npz, with probe_converged and probe_shuffled_converged true.
The gate checks these files and identities; test suites/review establish metric
correctness. Evidence is not a cryptographic certificate of research truth.

Aggregate repeats then seeds then participants equally. Withhold cohort means
unless all required tasks/fits are valid; retain failed/incomplete/negative rows.
Write Markdown and CSV tables for primary/secondary contrasts, per-subject scores,
clean train–validation gaps, probes, and robustness. Never emit test metrics.
Session 10 may conclude that none of the candidates warrants further work.

## Method sources

These sources motivate designs, not expected scores under our protocol:
[EEGNet](https://arxiv.org/abs/1611.08024),
[FBCNet](https://arxiv.org/abs/2104.01233), and
[EEG Conformer official implementation](https://github.com/eeyhsong/EEG-Conformer).
Our fixed local power and compact Transformer settings are proposed adaptations.
