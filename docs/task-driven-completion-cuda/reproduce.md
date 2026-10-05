# Check or reproduce the learned covariance results

Start here when reviewing the shared branch `research/baseline-accuracy`.
The completed study has nine participants, three seeds, five completion strategies,
54 supervised completer fits and 2,835 saved prediction cells. The classifier
is a frozen historical EEGNet–Transformer. All reported scores are validation
scores; the same degraded validation conditions selected completion checkpoints.

## 1. Check the saved results on any CPU

Clone this branch and run the verifier. This route needs Python 3.12+ and NumPy;
it needs neither PyTorch, CUDA, nor EEG recordings. It does not train anything.

```bash
git clone --branch research/baseline-accuracy https://github.com/georgiipr/agfl-inm.git
cd agfl-inm
python3 -m venv .review-venv
.review-venv/bin/python -m pip install numpy
AGFL_PYTHON="$PWD/.review-venv/bin/python" bash scripts/show_completion_results.sh --verify
```

The verifier checks archive/source/task hashes, partition IDs, masks, all saved
probabilities and metrics, repeat → seed → participant aggregation, and the
reported bootstrap intervals. It reads numeric NPZ arrays with pickling disabled;
it does not load PyTorch checkpoints. Expected output includes:

| Completion | 16 retained BA | 6 retained BA | Overall degraded BA |
|---|---:|---:|---:|
| Zero | 36.68% | 31.66% | 34.17% |
| Frozen Tucker | 40.12% | 31.91% | 36.01% |
| Learned Tucker | 41.87% | 35.04% | 38.45% |
| Frozen covariance | 58.27% | 40.87% | 49.57% |
| Learned covariance | 59.81% | 52.04% | 55.92% |

“16 retained” means six missing electrodes. Overall averages static/dynamic
retention at 16 and 6 electrodes; it excludes full input. Full-input BA is 64.87%
for every arm. The [report](../../evidence/learned-covariance-v2/report/summary.md),
[tables](../../evidence/learned-covariance-v2/report/cohort_conditions.csv), and
[method PDF](../concept/learned_covariance.pdf) are available without execution.

The numerical archive includes selected completion states, predictions, histories,
the two required historical backbone checkpoints per task, original split metadata,
independent GPU audit receipts, hidden-NaN diagnostics, and the preserved failed v1
attempt. It excludes GDF recordings and the unrelated historical fit artifacts.
Its [manifest](../../evidence/learned-covariance-v2/manifest.json) inventories every byte
hash. Verifying saved evidence is distinct from independently rerunning inference.

## 2. Prepare a fresh scientific reproduction

This path needs the nine original `A01T.gdf` through `A09T.gdf` recordings, CUDA,
GNU `timeout`, `flock`, and the recorded scientific package versions in
[`requirements-replay.txt`](../../evidence/learned-covariance-v2/requirements-replay.txt).
The original environment used Python 3.13.13 and an RTX 3060 Laptop GPU.
The requirements file records versions; it is not a promise that your package
index supplies those exact CUDA wheels. Obtain the matching environment/wheels
before running real fits. The preflight command reports a mismatch rather than
changing scientific dependencies or disabling compatibility checks.

The historical metadata binds the recording directory to
`/home/kalexu97/Projects/ml`. Preserve that path when reproducing: place the GDF
files there, or bind-mount their directory there inside your CUDA-enabled
container. No recordings are downloaded or redistributed by these scripts.
For example, with **an existing image containing the recorded environment**:

```bash
docker run --rm -it --gpus all \
  --mount "type=bind,src=$PWD,dst=/home/kalexu97/Projects/AGFL" \
  --mount "type=bind,src=/absolute/path/to/GDFs,dst=/home/kalexu97/Projects/ml,readonly" \
  -w /home/kalexu97/Projects/AGFL YOUR_MATCHING_ENVIRONMENT_IMAGE bash
```

Inside that environment, set `AGFL_PYTHON` to its Python executable. These examples
use the project virtualenv. Unpacking refuses to overwrite any different existing
file. Repeating it on identical files is safe.

```bash
export AGFL_PYTHON="$PWD/.venv/bin/python"
bash scripts/show_completion_results.sh --unpack

"$AGFL_PYTHON" scripts/completion_bundle.py prepare-config \
  --data-dir /home/kalexu97/Projects/ml \
  --config-out configs/task-driven-completion-cuda-friend.local.json \
  --output "$PWD/results/task-driven-completion-cuda-friend-v1"

bash scripts/run_completion_experiments.sh --plan \
  --config configs/task-driven-completion-cuda-friend.local.json
bash scripts/run_completion_experiments.sh --preflight \
  --config configs/task-driven-completion-cuda-friend.local.json
```

The new configuration changes only paths/name and creates its own source/config/
package identity. Original evidence is immutable. Historical package/source/data
checks remain in force. Another GPU may produce different numerical trajectories;
the new study records the actual GPU and must pass its CPU/CUDA comparison.
The original task identity also records the original GPU UUID: do not rewrite
archived metadata to make an original-task GPU audit pass on another machine.

## 3. Synthetic checks, pilot, then cohort

First run the structural and actual GPU tests; a missing GPU is a failure, not a skip.

```bash
"$AGFL_PYTHON" -m unittest discover -s tests/task_driven_completion_cuda -p test_protocol.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$AGFL_PYTHON" -m unittest discover \
  -s tests/task_driven_completion_cuda -p test_gpu.py -v

bash scripts/run_completion_experiments.sh --smoke \
  --config configs/task-driven-completion-cuda-friend.local.json \
  --output-dir "$PWD/results/task-driven-completion-cuda-friend-smoke"

bash scripts/run_completion_experiments.sh --pilot \
  --config configs/task-driven-completion-cuda-friend.local.json
```

The pilot fits A01/seed0, independently replays all 105 cells, and records an
operational acceptance receipt. Accuracy is never the advancement gate. Read its
logs, then launch the remaining tasks explicitly:

```bash
bash scripts/run_completion_experiments.sh --cohort \
  --config configs/task-driven-completion-cuda-friend.local.json
```

Fits have a two-hour timeout; audits have a 20-minute timeout. Each fit is followed
by independent selected-state replay. Any error/timeout stops the sequence, keeps
the output/logs, and prevents automatic retry. Logs and exit codes are under
`.session-runs/completion-reproduction/<config-hash>/`. A lock prevents two copies
of this launcher running together. Keep the foreground session alive, or launch
the explicit cohort command through your own terminal/session manager.

`--dry-run` prints the task commands without training or writing files. It does
not claim that runtime readiness/pilot checks have passed. Failed/partial tasks
cannot be retried in place; investigate first, then use a new config/output identity.
Do not tune settings based on this cohort and reuse the same study identity.

## 4. Inspect and audit your new results

The cohort command automatically writes its report after all 27 tasks pass.

```bash
cat results/task-driven-completion-cuda-friend-v1/report/summary.md
cat results/task-driven-completion-cuda-friend-v1/report/cohort_conditions.csv
bash scripts/run_completion_experiments.sh --audit \
  --config configs/task-driven-completion-cuda-friend.local.json
```

`--audit` reruns saved models without fitting. `--summarize` verifies and rebuilds
an existing report, but refuses to overwrite an existing launcher report attempt
log. No command silently resumes failed work. Different source, packages, hardware
or paths may change identity; that is not permission to relabel old results.

The gains are exploratory: checkpoint selection reused validation, epoch zero was
eligible, and no independent test/session confirmation is claimed. The frozen
covariance control already performs strongly; preserve its comparison and negative
Tucker-versus-covariance effects when discussing the results.
