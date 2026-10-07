# Published-checkpoint covariance runbook

**Runtime update:** the first real attempt stopped on a CUDA compiler-discovery
failure. Use the explicit environment and fresh output documented in
[runtime recovery](runtime-recovery.md). The original output remains immutable.

This study evaluates static electrode loss for the admitted EEG-ATCNet run-1
checkpoints. Its interpretation is **exploratory**: checkpoint provenance is
grade B, E-guided checkpoint selection is not conclusively excluded, and prior
project exposure to E outcomes remains unknown. Reports preserve those facts
and do not make a confirmatory claim.

Use the isolated runtime `.venv-published-covariance/bin/python`, Python 3.11,
TensorFlow 2.15.1, NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2 and h5py
3.10.0. The complete installed package freeze is
[`dependencies-lock.txt`](dependencies-lock.txt). It includes TensorFlow CUDA
12.2/cuDNN 8.9 wheels. The lock records installed packages; GPU driver, device,
and CUDA visibility remain host properties. `preflight` prints the current
runtime observation; the fresh study's `identity.json` records it when
`reference` initializes the output. Do not modify the existing `.venv`.

## Readiness and native T references

First check the dependency-light task declaration. On the allocated GPU host,
run preflight and the native synthetic smoke before any real fitting. The smoke
uses generated `[N,22,1125]` values and generated T-only IDs, executes all nine
actual checkpoints, performs an A01 covariance update, saves and reloads its
selected state, and writes only into a fresh directory outside `results/`.
The launcher refuses an occupied smoke output and records its process receipt.

```bash
python -m published_covariance plan
.venv-published-covariance/bin/python -m published_covariance preflight --device cuda:0
bash scripts/run_published_covariance.sh smoke --native --device cuda:0 --output /tmp/published-covariance-native-smoke-SESSION
```

Replace `SESSION` with a new unique name whose directory does not exist. Review
`native-smoke.json`, `native-smoke.npz`, and the supervisor's independent
`acceptance_cuda.py` result. The coding worker had no visible GPU; the supervisor
subsequently passed the native CUDA gate on an RTX 3060 Laptop GPU in 25.316
seconds (receipt:
`.session-runs/published-checkpoint-covariance/06/native-cuda.json`). Smoke
artifacts are software evidence only and must never enter a real report.

Choose a fresh real output identity such as
`results/published-covariance-v1`. If source, configuration, plans, assets,
packages, or runtime flags change after any output is created, use a new output
directory. Real outputs must be direct children of `results/` named
`published-covariance-*`. The Bash launcher holds `.session-runs/accuracy.lock`
and preserves command, stdout, stderr, start/end time and exit status under
`.session-runs/published-checkpoint-covariance/launcher/`. It uses GNU timeout,
TERM and then KILL after 20 seconds: each real fit has a 3,600-second ceiling;
references, task audits, evaluation tasks, review, final audit and summary have
600-second ceilings. A failed or timed-out child is not retried or erased.

Create native full-input T references before the pilot so source parity is
reviewed before covariance fitting:

```bash
bash scripts/run_published_covariance.sh reference --output results/published-covariance-v1 --device cuda:0
bash scripts/run_published_covariance.sh pilot --output results/published-covariance-v1 --device cuda:0
bash scripts/run_published_covariance.sh review-pilot --output results/published-covariance-v1 --device cuda:0
```

The pilot fits A01 seeds 0, 1 and 2 on T only. Review independently replays
their split, train-only covariance initialization, validation selection and
frozen classifier behavior. Inspect pilot wall time and peak device memory,
then record the operational capacity decision before starting the remaining
fits. The fixed launcher ceiling is one hour per task; a timeout is a recorded
failure and is never addressed by changing epoch, optimizer, or selection
settings. The pilot review receipt gates all other participants.

## Fitting, evaluation and reporting

After the pilot review, fit and audit the remaining 24 tasks, then seal all 27
selected T-only covariance states. The seal is required before any E evaluation.

```bash
bash scripts/run_published_covariance.sh train-cohort --output results/published-covariance-v1 --device cuda:0
bash scripts/run_published_covariance.sh evaluate-cohort --output results/published-covariance-v1 --device cuda:0
bash scripts/run_published_covariance.sh audit --output results/published-covariance-v1 --device cuda:0
bash scripts/run_published_covariance.sh summarize --output results/published-covariance-v1 --device cuda:0
.venv-published-covariance/bin/python plans/published-checkpoint-covariance/verify_real_report.py results/published-covariance-v1
```

The evaluator processes one subject/seed at a time, computes the three paired
arms under full, static-16 and static-6 masks, and runs an independent task audit.
The final audit requires all 27 fits, nine T references, 81 arm evaluations and
all 891 evaluation cells before aggregation. `verify_real_report.py` independently
recomputes the participant hierarchy and bootstrap interval from numeric
prediction files. Run `summarize` again to confirm byte-stable report output.

The comparison uses covariance seeds 0/1/2 for T splits, training order and
masks; these are not independent pretrained-classifier seeds. The pretrained
backbone and its normalization already saw all T trials, including the later
covariance-validation subset. No E data is read during fitting or checkpoint
selection. E is the held evaluation for this covariance procedure, but the
external checkpoint's selection history and earlier project E exposure remain
as disclosed in the contract.

Expected declaration: 27 covariance fits, zero backbone fits, 81 arm evaluations,
891 prediction cells, nine native full-input references. Each learned covariance
has 253 free parameters; each frozen classifier has 115,172 parameters.
Existing assets occupy about 788 MiB, the isolated runtime about 4.7 GiB, and
real outputs are estimated near 1.5 GiB. Check available space before the GPU
run; the session-06 worker observed about 3.1 GiB free. Keep sufficient headroom
for logs and temporary files. Do not delete existing assets, outputs, or receipts
to make room.

## Synthetic software lifecycle

This complete generated-data lifecycle checks public pilot, review, fit seal,
T reference, evaluation, audit and summary paths on CPU. Choose a fresh output
outside `results/`; never use the resulting artifacts as measurements. The
native smoke above is a separate GPU test and does not consume this output.

```bash
.venv-published-covariance/bin/python -m published_covariance pilot --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
.venv-published-covariance/bin/python -m published_covariance review-pilot --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
.venv-published-covariance/bin/python -m published_covariance train-cohort --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
.venv-published-covariance/bin/python -m published_covariance reference --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
.venv-published-covariance/bin/python -m published_covariance evaluate-cohort --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
.venv-published-covariance/bin/python -m published_covariance audit --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
.venv-published-covariance/bin/python -m published_covariance summarize --synthetic --device cpu --output /tmp/published-covariance-synthetic-SESSION
```

Replace `SESSION` with a fresh unique directory name. The synthetic output is a
software fixture with generated labels, a toy classifier and reduced training
budgets. The real-report verifier intentionally rejects it.
