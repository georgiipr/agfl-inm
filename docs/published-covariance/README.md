# Validate the published ATCNet covariance experiment

This is a separate reproduction of the experiment introduced in
[`research/baseline-accuracy` at revision `516922d`](https://github.com/georgiipr/agfl-inm/tree/516922dff6d1fe9879e774ace76bb7366dcc5f28/published_covariance).
It retains the frozen official ATCNet run-1 checkpoints for all nine
participants. It does not retrain EEGNet, replace the existing six INM variants,
or reproduce the original AGFL manuscript's MHA-versus-AGFL experiment.

## Goal and fixed protocol

The goal is to independently reproduce full-channel classification and the
effect of frozen versus learned covariance under the same published protocol.
Compare three arms: zero filling, frozen covariance and learned covariance.
Evaluate 22 channels once and static random 16/6-channel outages with five
repeats each. This is 27 covariance fits, zero backbone fits, 81 arm evaluations
and 891 saved prediction cells. Keep every participant, seed and outcome.

Each participant uses its own frozen classifier. Seeds 0/1/2 vary covariance
splits, mask draws and batch order; they are not classifier initialization seeds.
The classifier, BatchNorm and dropout remain frozen. Full-channel completion
is bypassed exactly, and all three arms must produce identical full predictions.

1. Verify upstream source/checkpoint and recording SHA256 identities. Check the
   runtime, all checkpoint tensor bytes, MHA equations, GPU gradients and a tiny
   generated-input covariance fit. These checks run on the experiment GPU.
2. Load native T-session trials, retain all 288 trials including artifacts, and
   reproduce the upstream crop and scaler. Create classwise 80/20 T splits:
   232 training and 56 covariance-validation trials.
3. Fit the 22x22 second moment on the covariance-training subset only. Fit the
   learned 253-coordinate Cholesky covariance with the classifier frozen.
   Select it using degraded T-validation balanced accuracy, then lower log loss,
   then the earlier epoch. Preserve epoch zero as an eligible checkpoint.
4. Audit and seal all 27 selected covariance states before opening E-session
   signals or labels for evaluation. E outcomes never select covariance weights.
5. Save all E masks, trial identities and probabilities, replay every task,
   aggregate repeats then seeds then participants, and independently recompute
   the report from saved probabilities.

The optimizer and mask protocol remain those of the research branch. Covariance
training uses Adam with learning rate 0.001, batch size 32, global gradient
clipping at 1, at most 100 epochs, at least 10 epochs, and patience 20. The loss
is classification cross entropy plus 0.1 hidden-sample MSE plus 0.0001 mean
squared displacement of Cholesky coordinates from initialization. Training masks
sample 22/16/6 retained electrodes. Subset pools are reserved by partition.
See [configuration](../../configs/published-covariance.json),
[import provenance](import-origin.json) and [the upstream model](https://github.com/Altaheri/EEG-ATCNet/tree/65162fb359ea46a2f62c885a9987247ab491ee7a).

## Interpretation

This is an exploratory reproduction. The external checkpoints have incomplete
selection provenance: their historical selection using E is not conclusively
excluded. The original classifier and scaler saw all T trials, including the
later covariance-validation subset. That historical scaler is replayed to keep
the pretrained model's input contract; the newly fitted covariance sees only
its declared training subset. Prior project exposure to E outcomes is unknown.
Reproducing a score establishes reproducibility under this protocol; it does
not resolve the external checkpoints' history or establish an untouched-test
generalization claim. These disclosures remain in the generated report.

## Existing runtime

Use the existing GPU TensorFlow environment; this workflow installs no packages.
Python 3.7 or newer, NumPy with PCG64 (1.17 or newer), SciPy, scikit-learn and
h5py must already be available. The existing PyTorch environment and its
requirements remain separate.

The reference branch ran TensorFlow 2.15.1. TensorFlow 2.2 lacks native Keras MHA.
For that environment, `tf_compat.py` implements only the published checkpoint's
fixed two-head, key-dimension-eight temporal self-attention, preserving its
eight projection tensors and their H5 order. Unsupported attention options fail.
Newer TensorFlow uses native Keras MHA.

Before fitting, preflight checks all loaded checkpoint tensor bytes and compares
the compatibility attention outputs against independent NumPy equations using
the actual checkpoint weights. When native MHA exists it also compares native
and compatibility outputs. On TensorFlow 2.2 the upstream reference and wrapper
share this compatibility attention, so their mutual parity is not an independent
test of MHA; the report states that limitation. Numerical runtime compatibility
must be established on the cluster. No local model execution is required.

## Dataset and launch

The dataset is [BCI Competition IV, data set 2a](https://www.bbci.de/competition/iv/).
This experiment needs native structured T and E MAT recordings, not label-only
evaluation MAT files. Preparation downloads the 18 pinned files from the
[BNCI dataset page](https://bnci-horizon-2020.eu/database/data-sets) and the
upstream source/checkpoints. It checks every digest in
[recordings-sha256.txt](recordings-sha256.txt) and the upstream origin manifest.
Honor the dataset's usage and citation terms. Preparation needs internet access
and approximately 0.8 GB for recordings. Downloaded assets are ignored by Git.

Prepare assets on a machine with internet access before allocating an experiment
GPU. Downloads use IPv4 without changing TLS verification or the pinned file
hashes. Native recordings are fetched directly from the TU Graz dataset server,
with three attempts for network failures and progress printed during transfers.
Completed files are verified and reused. Earlier verified origin receipts are
preserved when their recording hashes match. GPU experiment stages use local
assets only; a missing asset stops the job instead of starting a network
download. The same prepared bundle is shared
by the covariance, Q1 and Q2 experiments.

Run from the repository root in the existing TensorFlow environment:

```bash
python -m published_covariance prepare-assets
python -m published_covariance prepare-assets --offline
python -m published_covariance plan
python -m published_covariance preflight --device cuda:0
python -m published_covariance smoke --native --device cuda:0 --output scratch/atcnet-native-smoke

output=results/published-covariance-validation-v1
python -m published_covariance reference --device cuda:0 --output "$output"
python -m published_covariance pilot --device cuda:0 --output "$output"
python -m published_covariance review-pilot --device cuda:0 --output "$output"
python -m published_covariance train-cohort --device cuda:0 --output "$output"
python -m published_covariance evaluate-cohort --device cuda:0 --output "$output"
python -m published_covariance audit --device cuda:0 --output "$output"
python -m published_covariance summarize --device cuda:0 --output "$output"
python -m published_covariance.verify "$output"
```

The pilot uses A01 and three covariance seeds as a software/data gate. Its
scores do not determine inclusion or change the protocol. `review-pilot`
verifies saved states automatically; no manual scientific approval is required.
Choose a new smoke path for another native smoke run. Changed source, settings
or runtime require a fresh study output suffix; completed incompatible studies
are never overwritten. The task CLI supports parallel scheduling while keeping
fit/evaluation barriers and shared writes serialized.

The final output includes `report.md`, `report.json`, `audit.json`, `identity.json`,
all selected fits/histories, native references, normalization, masks and saved
probabilities. The supplied launcher also retains GPU admission receipts and
generated-input smoke artifacts under `admission/`. The independent verifier reports completeness and recomputed
participant effects. Existing INM reports stay in their original directories.

## Attribution

The native ATCNet graph is adapted from Hamdi Altaheri's EEG-ATCNet, pinned at
`65162fb359ea46a2f62c885a9987247ab491ee7a`, copyright 2022 King Saud University.
The source is distributed under [Apache-2.0](LICENSE). The downloaded upstream
source retains its original license and attribution. Import provenance records
the source experiment revision and the original module hashes.
