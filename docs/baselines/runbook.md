# Baseline study runbook

This is the operational guide for the separate `inm.baselines` study. Its
scores are **unknown until experiments run**. A passing synthetic smoke checks
software paths only and does not provide EEG accuracy evidence.

The first full within-session cohort has now been reviewed. See
[results-review.md](results-review.md) for its measured outcomes and the
initialization correction. For new training, use
`configs/baselines-reproducible.json` with its fresh output directory. The
commands below describe the original study; its completed artifacts must stay
unchanged. Reports for the original config can still be regenerated.

## Environment and resolved paths

Use Python 3.12 or newer with the scientific packages from `requirements.txt`,
including PyTorch, NumPy, SciPy, scikit-learn, and MNE. Real neural tasks default
to CUDA; use a CUDA-enabled PyTorch environment and a working CUDA device. Keep
the environment tied to this checkout:

```bash
cd /path/to/AGFL
export AGFL_PYTHON="$PWD/.venv/bin/python"
"$AGFL_PYTHON" -c 'import numpy, scipy, torch, sklearn, mne; print("scientific imports passed", torch.cuda.is_available())'
```

The session-10 environment check used
`/home/kalexu97/Projects/AGFL/.venv/bin/python` (Python 3.13.13, then CPU PyTorch
2.14.1, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, MNE 1.13.2). No package
installation or GPU setup was performed as part of that review. For a new
environment, prepare Python and install the packages using the project
requirements and the PyTorch build that matches the target hardware before
running these commands. The workstation environment has since been updated to
CUDA PyTorch; see [environment.md](environment.md). Use the launcher to select
the project interpreter and check actual GPU access outside a restricted sandbox:

```bash
bash scripts/run_baselines.sh --check-cuda
bash scripts/run_baselines.sh --config configs/baselines.json --preflight
```

Paths in the shipped configs resolve relative to the config file, not the
launching directory:

| Config value | Resolved path in this checkout |
|---|---|
| `configs/baselines.json`: `data_dir: ../../ml` | `/home/kalexu97/Projects/ml` |
| `configs/baselines.json`: `output_dir: ../results/baselines-v1` | `/home/kalexu97/Projects/AGFL/results/baselines-v1` |
| `configs/baselines-cross-session.json`: `data_dir: ../../ml` | `/home/kalexu97/Projects/ml` |
| `configs/baselines-cross-session.json`: `output_dir: ../results/baselines-cross-session-v1` | `/home/kalexu97/Projects/AGFL/results/baselines-cross-session-v1` |

The within-session config expects `A01T.gdf` through `A09T.gdf` in the data
directory. Cross-session additionally needs each E-session recording and
official E labels supplied through an explicit `external_labels_dir`; labels
are never inferred. The current workstation had no recordings at the resolved
data path when session 10 was reviewed.

## Preflight, plan, and isolated smoke

From the repository root:

```bash
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines.json --plan
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines.json --preflight
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines.json --smoke
```

The plan is 27 participant/seed tasks and 108 baseline fits. Preflight exits
nonzero and names missing packages, recordings, or explicit cross-session
labels. Smoke uses synthetic inputs, CPU, at most two epochs, and writes beside
the real output directory to a source-identity-suffixed
`baselines-v1-synthetic-smoke-*` directory. Do not interpret or aggregate those
scores as real results. A smoke directory is intentionally isolated and
single-use; run it with a fresh output/config path if its artifacts already
exist.

## Pilot and resume

After preflight is ready and a CUDA environment is available, run task 0, which
is participant A01, seed 0. It fits the four declared arms on matched splits.
This is the real pilot prerequisite for later ablation work:

```bash
bash scripts/run_baselines.sh --config configs/baselines.json --task-index 0 --device cuda
```

Use `--device cpu` only when intentionally running the real task on CPU. A
requested CUDA device fails if CUDA is unavailable; task execution never falls
back to CPU or substitutes synthetic data for missing GDF inputs.

Completed arms are recognized and reused when the same task command is run
again with unchanged config, source, input data, and output directory. Inspect
`artifacts/A01_seed_0/ARMS/<arm>/failure.json` after a failed fit. Incomplete
arm artifacts are preserved and rejected for reuse; do not delete or overwrite
them. Retry a failed/incomplete fit in a fresh output directory with a copied
config whose `output_dir` points there. A changed config, package source, or
recording also requires a fresh directory.

Selected neural checkpoints are saved at:

```text
results/baselines-v1/artifacts/A01_seed_0/ARMS/eegnet_reference__full/checkpoint.pt
results/baselines-v1/artifacts/A01_seed_0/ARMS/masked_eegnet__full/checkpoint.pt
results/baselines-v1/artifacts/A01_seed_0/ARMS/masked_eegnet__mixed/checkpoint.pt
```

Each neural arm also saves `history.json` and `result.json`; the covariance arm
saves `model.npz` and `result.json`. The task and study manifests record
identities and checksums. Checkpoints contain the selected validation epoch,
constructor settings, weights, normalization stats, channel/class order, and
study provenance.

## Reporting and larger runs

Rebuild reports from the declared output directory after tasks finish:

```bash
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines.json --summarize-only
```

Read `results/baselines-v1/report/summary.md` and `progress.json` first. Missing,
invalid, synthetic, or identity-incompatible records prevent a complete cohort
mean. A complete cohort requires every configured participant, seed, and arm.
Aggregation averages mask repeats, then seeds within participant, then
participants equally.

Only after the A01/seed-0 pilot and its histories/report have been inspected,
run the remaining task indices to complete the nine-participant, three-seed
matrix. The following serial loop is a template and was not launched in this
review:

```bash
for task in $(seq 0 26); do
  "$AGFL_PYTHON" -m inm.baselines --config configs/baselines.json --task-index "$task" --device cuda || break
done
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines.json --summarize-only
```

This session did not schedule GPU work or run a real pilot/full study.

## Cross-session study

Inspect its declared matrix first:

```bash
"$AGFL_PYTHON" -m inm.baselines --config configs/baselines-cross-session.json --plan
```

The shipped config leaves `external_labels_dir` null to make the prerequisite
visible. Copy it to a separately named config, set that field to the supplied
official E-label directory, and verify the resolved path and preflight before
execution:

```bash
"$AGFL_PYTHON" -m inm.baselines --config /path/to/baselines-cross-session-local.json --preflight
```

Cross-session train and validation use T only, with E exclusively held out for
test. Keep its output directory distinct from within-session results and never
combine protocol reports.
