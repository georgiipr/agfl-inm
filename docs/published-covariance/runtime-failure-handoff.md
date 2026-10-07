# Runtime failure diagnosis: A05-s1 E evaluation

## Finding

The failed child was the A05 seed-1 evaluation command recorded in
`.session-runs/published-checkpoint-covariance/launcher/20261007T010348Z-215256-A05-s1-E-evaluate.command.json`.
It exited 134. Its stderr reaches CUDA device creation and cuDNN initialization,
then TensorFlow's XLA GPU LLVM backend aborts while opening
`tensorflow/python/platform/../../libtensorflow_framework.so.2/../../nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc`
with `Could not open input file: Not a directory`. The immediately preceding
warning, `Start cannot spawn child process: No such file or directory`, is
consistent with a compiler subprocess lookup problem. The repeated cuDNN/cuFFT/
cuBLAS registration warnings and CPU feature messages occur during startup; they
do not explain this fatal file-open error. The wrapper recorded exit 134.

This is a CUDA compiler asset-discovery failure, not evidence of a bad EEG trial,
checkpoint, covariance state, or model/optimizer setting. The error path passes
through `libtensorflow_framework.so.2` as if it were a directory before walking
to `nvidia/cuda_nvcc`; that component is a shared-object file. The isolated
runtime actually contains the expected CUDA compiler package at
`.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc/`:

- `bin/ptxas` is an executable file.
- `nvvm/libdevice/libdevice.10.bc` is a readable LLVM IR bitcode file.
- The paths resolve to ordinary files under this checkout's virtualenv, not
  dangling links or missing package contents.

The runtime freeze pins TensorFlow 2.15.1 and `nvidia-cuda-nvcc-cu12` 12.2.140.
Earlier fit stderr files show XLA initialized and compiled clusters, and the
session-06 generated native CUDA smoke passed. Those checks establish that the
GPU and some XLA compilation paths worked; they do not guarantee discovery for
every generated operation. The failed E-evaluation log demonstrates that this
compiler path is not reliably resolved by the current implicit search. The
native smoke did not explicitly exercise a chosen libdevice-dependent math
graph, so it should be paired with the dedicated generated compiler probe below.

The related `Start cannot spawn child process` warning could be benign for one
subprocess or could indicate that `ptxas` was also missed. The proposed explicit
CUDA data root gives XLA the standard `bin/ptxas` and `nvvm/libdevice` layout in
one setting. No `PATH` change is needed initially: use `PATH` only if the probe
with that root still reports that it cannot spawn/find `ptxas`, and treat any
such additional environment change as a further identity change.

## Minimal repair and synthetic acceptance

For every process that imports TensorFlow, set the following before startup:

```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kalexu97/Projects/AGFL/.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc
```

This points at the package root, where XLA expects `bin/ptxas` and
`nvvm/libdevice/libdevice.10.bc`. It avoids package downloads, package
installation, system CUDA changes, TensorFlow changes, and source changes. The
existing runtime is left intact. Check these assets before compiling:

```bash
test -x .venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas
test -r .venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
```

On the allocated CUDA host, first run the supervisor's generated-only XLA
compiler probe in two fresh processes. It uses `jit_compile=True` on float64
`lgamma`, `erf` and `sin`, compares outputs and gradients with CPU results, and
prints hashes for both compiler assets. It reads no EEG data and evaluates no
real outcomes:

```bash
XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kalexu97/Projects/AGFL/.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_runtime_compiler.py
XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kalexu97/Projects/AGFL/.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_runtime_compiler.py
```

Then run the independent all-nine native forward/backward and one-epoch
generated covariance smoke in its fresh temporary output. The environment is
inherited by the public launcher and the Python child:

```bash
XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kalexu97/Projects/AGFL/.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_cuda.py
```

Accept the runtime repair only if both independent compiler-probe processes
exit zero, report GPU placement and finite output/gradient agreement, and the
all-nine smoke exits zero with frozen state and saved-state replay checks intact.
Preserve stdout, stderr and each exit in fresh supervisor-owned receipts. A
successful probe directly establishes discovery for the tested libdevice math
and `ptxas` path; the all-nine smoke separately checks the actual classifier and
covariance forward/backward route. Neither is a real fit or E evaluation. A
single repeated pass cannot prove that a low-level process-spawn failure is
impossible, but two fresh compiler processes plus the full generated smoke give
useful evidence that discovery is stable for this runtime and host. Keep any
failure and do not retry a real task as a diagnostic.

## Identity and resumption

The setting does not change the declared classifier, covariance equations,
optimizer, training schedule, masks, or evaluation protocol. It supplies XLA's
compiler-data directory so the existing pinned runtime can find its installed
compiler assets. It is therefore an operational compiler-discovery repair, not
a model or optimizer change.

The study's canonical runtime identity records `XLA_FLAGS` (along with package
versions, device information, and other runtime flags). Setting it changes that
identity even though the package freeze and scientific settings stay fixed. The
contract requires a new output directory after runtime changes, and the current
study code rejects an existing output whose identity differs. Do not patch,
override, or bypass that comparison. The original `results/published-covariance-v1`
cannot be resumed under the proposed flag, and a retry under the old implicit
search is not an acceptable repair: this failure is recorded and the launcher
receipt explicitly stopped advancement with no retry authorized.

If the supervisor accepts the compiler probe and chooses to continue with the
explicit flag, use a fresh study identity and refit all 27 covariance tasks
under it before any successor E evaluation. The fits do not become scientifically
different because the model or optimizer changed; they must be repeated because
the recorded runtime identity changed and prior fit artifacts are identity-bound.
Use the same frozen scientific declaration and seeds. Do not copy prior fit
states, task audits, evaluation cells, or seals into the new output.

Preserve the old output and receipts as read-only historical evidence: they show
that all 27 old-identity covariance fits and 13 E task audits completed, while
A05-s1 produced zero task cells and no task receipt. The prior partial exposure
to E remains disclosed in any successor report; a fresh output does not erase
it. Old outputs can support an audit trail and the failure diagnosis, but cannot
be counted toward the successor's required fits, references, or evaluations.
Pinned source, checkpoint and recording assets may be reused as immutable inputs
only after their existing origin/hash manifests are rechecked and their hashes
are recorded in the new identity. The accepted session-06 smoke receipt can be
cited as prior generated-runtime evidence, but it does not substitute for the
new explicit-flag compiler probe or native smoke.

## Scope and evidence reviewed

This diagnosis used the failure stderr and launcher command/exit receipt, the
session-06 integration handoff and CUDA receipt, the runtime freeze and package
layout, the study's runtime identity handling, and the published covariance
contract and acceptance requirements. No accuracy outcomes were inspected. No
GPU command, real task, package installation, source/runtime edit, or output
mutation was performed for this diagnosis.
