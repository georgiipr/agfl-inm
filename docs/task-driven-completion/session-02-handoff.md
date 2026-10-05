# Session 02 handoff: differentiable paired completion

Completed 2026-10-05 within the supervisor's bounded session. Read root
`AGENTS.md`, project orientation, the contract/session, session-01 handoff and
accepted receipt. Changed only the four files authorized for this session:

- `inm/task_driven_completion/completion.py`
- `inm/task_driven_completion/adapters.py`
- `tests/task_driven_completion/test_models.py`
- `docs/task-driven-completion/session-02-handoff.md`

Both Tucker arms reuse the existing differentiable `observed_ridge_core`
without modifying it: float64 solves with penalty `n_observed * 250 * 0.001`.
Only missing entries receive reconstructed values. Learned U/V have unit
columns after `post_step()`. Both covariance arms use identical packed free
lower-triangle parameters, softplus diagonal plus `1e-6`, float64 conditional
solves and ridge `0.001 * mean(diag(Sigma))`. Inverse softplus initialization
reconstructs the training second moment plus `1e-6 I`. Frozen parameters are
buffers; learned parameters are Parameters. Initial anchor buffers are separate
copies and remain part of persisted numeric state.

The Transformer adapter preserves original flags and whole-trial EEGNet layers.
`adapter.train()` enables completion training while every backbone module remains
eval and every backbone parameter remains frozen. Forward rejects externally
reenabled backbone training/gradients and mismatched strategy/completer families
or learned/frozen status. Gradients flow through the frozen classifier into
completion factors. Full masks bypass completion exactly; zero strategy uses
the original backbone path exactly. No completion gradient arises from full
input, including the full examples within a mixed batch.

## Session 03 APIs

Import numerical APIs directly from `completion` and `adapters`; session-01
dependency-light `__init__.py` remains unchanged.

```python
initialization, factor_history = fit_initialization(train_x, task_seed)
models = paired_completers(initialization)
completer = models[strategy]
adapter = CompletionAdapter(frozen_transformer, strategy, completer)
completed = completer(train_x, mask)  # .complete is equivalent
penalty = completer.anchor_penalty()
reconstruction = reconstruction_loss(completed, train_x, mask)
# After the caller's optimizer.step():
completer.post_step()
arrays = completion_state_arrays(completer)
restored = restore_completer(strategy, arrays)
```

`fit_initialization` is CPU-only and accepts normalized training inputs
`[N,22,4,250]`, task seed and optional `partition='train'`; other partitions are
rejected. It fixes Tucker fitting to 30 epochs, lr .01, batch64 and seed
`task_seed + 810001`. No label/validation arguments exist. The caller must still
verify split provenance, since a tensor cannot prove its origin. Returned
initialization has tensors `U`, `V`, `second_moment` and scalar int64
`training_trials`. History is the existing Tucker fitter's 30 numeric epoch
rows. `paired_completers` builds all five strategies from this single fit,
with exact paired states and outputs but independent storage.

Use the calibrated float32 Tucker factors with float32 historical backbone
input. Covariance free parameters intentionally remain float64 even with
float32 input; do not apply `.float()` indiscriminately to the adapter.
`scoped_cpu_rng(seed, threads=1)` is reusable for future model/data scopes;
it restores Python, NumPy, CPU Torch RNG and Torch thread count on success and
exceptions. CUDA remains unverified.

`anchor_penalty` averages squared displacement over all free parameters (U/V
elements combined for Tucker). The trainer owns weights CE + .1 reconstruction
+ .0001 anchor, mask pairing, optimizer configuration and selection. The
reconstruction helper selects hidden entries before subtraction. Empty targets
give differentiable zero; if full bypass has no input graph this is a standalone
zero scalar leaf, deliberately unconnected to factors. **Skip wholly full
batches before optimizer updates**, including anchor-only updates.

`completion_state_arrays` returns copied numeric NumPy arrays including initial
anchors. `restore_completer` validates exact keys, dimensions, dtypes, finite
values, Tucker column norms and frozen-state/anchor equality; it rejects unknown
strategies and returns the proper learned/frozen model. The runner must own NPZ
files with `allow_pickle=False`, checksums, selected epoch, factor history,
initialization metadata, identity verification and resume rules. State helpers
do not claim artifact authenticity or authorize resume.

## Actual checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python plans/task-driven-completion/acceptance.py 02
git diff --check
.venv/bin/python -c 'import ast; from pathlib import Path; paths = [Path("inm/task_driven_completion/completion.py"), Path("inm/task_driven_completion/adapters.py"), Path("tests/task_driven_completion/test_models.py")]; [ast.parse(path.read_text()) for path in paths]; print("Parsed three scoped Python files")'
```

Final mandatory acceptance: **22 tests passed, no skips**, 1.135 seconds.
Whitespace and scoped syntax checks passed. The inherited even-kernel
`padding='same'` convolution emits its existing PyTorch warning; no check failed.
Tests independently assemble the explicit observed design, verify normalized
ridge and repeated-column stability, compare central finite-difference gradients
for both U/V and covariance diagonal/off-diagonal free parameters, check exact
paired initialization, hidden NaN/Inf invariance, preserved observations, window
locality, full/zero equivalence, original flags and zero hidden-input gradients.
Classification CE alone gives finite nonzero gradients for each learned factor
family and raw input through the actual historical Transformer. Optimizer updates
change completers while all backbone tensors, BN buffers and gradient flags stay
unchanged. Tests also cover calibration repeatability and RNG restoration on
failure, post-step normalization, anchor formulas, empty reconstruction targets,
numeric state reconstruction and malformed-state rejection.

No real recordings or historical checkpoint arrays were read by this worker;
all calibration and gradient checks used synthetic CPU tensors. No real fitting,
degraded evaluation, study output or empirical benefit is claimed. Session 03
still owns deterministic training, partition-disjoint masks, epoch-0 selection,
artifact/identity-safe execution and reporting. Session 04 owns public integration
and regressions. Supervisor acceptance and the separate real pilot/cohort gates
remain required.
