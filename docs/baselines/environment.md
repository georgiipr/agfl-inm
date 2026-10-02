# Environment for implementation sessions

Sessions 03 onward require the scientific packages in `requirements.txt`, even
though their acceptance tests use synthetic CPU data. The session runner chooses
`AGFL_PYTHON`, then the repository's `.venv/bin/python`, then `python3`.
An explicit `--python` takes precedence over all three.

From the repository root, resume using the project-local environment:

```bash
bash scripts/run_accuracy_sessions.sh --from 03
```

Alternatively, set the interpreter for subsequent runner commands in this shell:

```bash
export AGFL_PYTHON="$PWD/.venv/bin/python"
bash scripts/run_accuracy_sessions.sh --from 03
```

The `.venv` directory is ignored by Git and isolated from existing Conda
environments. It now uses CUDA-enabled PyTorch and can also run the CPU tests.
Real experiments require the official recordings; synthetic acceptance tests
do not need recordings. An experiment preflight can therefore still report
missing data after the Python environment is ready.

Run GPU checks and real tasks from a host terminal with GPU access. A restricted
sandbox may hide the GPU even when the host driver is working. When using an
assistant, request outside-sandbox execution of the GPU command; the scripts
do not change sandbox permissions or silently fall back to CPU.

```bash
bash scripts/run_baselines.sh --check-cuda
bash scripts/run_baselines.sh --config configs/baselines.json --preflight
# Only after recordings are present; real tasks default to CUDA:
bash scripts/run_baselines.sh --config configs/baselines.json --task-index 0
```

The baseline launcher uses `AGFL_PYTHON` or the repository `.venv`, changes into
the repository, and forwards CLI arguments literally. `--check-cuda` reports
the interpreter/build/device and runs a small GPU forward/backward computation.
It distinguishes CPU-only PyTorch from a CUDA build that cannot access a device.
`--device cpu` explicitly selects CPU for real tasks; `--smoke` stays on CPU.

To provision the same kind of environment in a fresh checkout (Python 3.12+):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy, scipy, torch, sklearn, mne, tqdm; print("Scientific imports passed")'
```

A blocked session has no completion receipt. Retrying session 03 preserves its
previous logs, rechecks sessions 01–02, and starts a fresh attempt. Do not delete
receipts, edit protected plans, or mark the blocked attempt complete to change
interpreters.

## Local verification (2026-10-01)

The initial project `.venv` used Python 3.13.13 with NumPy 2.5.3, SciPy 1.18.1,
PyTorch 2.14.1+cpu, scikit-learn 1.9.1, MNE 1.13.2, and tqdm 4.70.1.
Its `pip check` passed. A synthetic numerical check passed SciPy bandpass filtering,
PyTorch loss/backward, a scikit-learn classifier fit, MNE RawArray construction,
and imports of the existing project loader and split helpers. The runner
rechecked sessions 01–02 successfully using this interpreter. Sessions 03–10
subsequently completed in that environment.

The CUDA update replaces only the PyTorch build and installs its runtime
dependencies, using the matching 2.14.1 release from the official `cu130` index.
The host NVIDIA driver is 595.91.07; `nvidia-smi` reports CUDA driver support
through 13.2 and an RTX 3060 with 6 GiB VRAM. The PyTorch runtime is 13.0;
the driver-supported version and wheel runtime need not be identical.
Consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
when provisioning other machines.

Post-update checks passed: `pip check`, 50 baseline tests, and 15 automation
tests. Outside the sandbox, `--check-cuda` passed on the RTX 3060 Laptop GPU.
Both reference and mask-conditioned EEGNet also completed one synthetic CUDA
batch of 32 trials, including convolution, backward, AdamW, and hidden-NaN
isolation (about 138 MiB peak allocated memory in that check). This is a runtime
check, not an EEG accuracy measurement.

The runner's environment-default change required a reviewed refresh of existing
receipt fingerprints. Original receipts and runner snapshots are preserved in
`.session-runs/accuracy/cuda-environment-update/`; all ten handoffs are unchanged.
The protected scientific plans and acceptance checker were not changed.
