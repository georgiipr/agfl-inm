"""Matched raw completion models; calibration accepts training data only.

The caller owns split isolation. No labels, validation arrays or test arrays are
accepted by calibration. All dimensions/settings are the fixed study declaration.
"""
from __future__ import annotations

from contextlib import contextmanager
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inm.supervised_tucker.models import observed_ridge_core
from inm.tensor_attention import Tucker2
from .protocol import STRATEGY_IDS


def validate_input(x: Tensor, mask: Tensor) -> Tensor:
    """Select observations before arithmetic, allowing arbitrary hidden values."""
    if (not isinstance(x, Tensor) or x.ndim != 4 or x.shape[0] < 1 or
            tuple(x.shape[1:]) != (22, 4, 250) or
            x.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Raw input must be nonempty float32/float64 [B,22,4,250]")
    if (not isinstance(mask, Tensor) or mask.dtype != torch.bool or
            mask.shape != x.shape[:3] or mask.device != x.device):
        raise ValueError("Availability must be Boolean [B,22,4] on the input device")
    if not mask.any(dim=1).all().item():
        raise ValueError("Every window requires at least one observed channel")
    safe = torch.where(mask[..., None], x, torch.zeros_like(x))
    if not torch.isfinite(safe).all().item():
        raise ValueError("Observed samples must be finite")
    return safe


@contextmanager
def scoped_cpu_rng(seed: int, *, threads: int = 1):
    """Restore Python/NumPy/CPU Torch RNG and thread count, even on failure."""
    if type(seed) is not int or seed < 0 or type(threads) is not int or threads < 1:
        raise ValueError("seed must be nonnegative and threads positive integers")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    thread_count = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            torch.set_num_threads(threads)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(thread_count)


def fit_initialization(train_x: Tensor, task_seed: int, *, partition: str = "train"):
    """Return (numeric initialization, 30-epoch Tucker history), once per task.

    Only normalized TRAINING trials belong here. The explicit partition guard
    catches accidental caller misuse but cannot infer the origin of a tensor.
    Calibration is CPU-only; fixed seed is task_seed + 810001.
    """
    if partition != "train":
        raise ValueError("Completion calibration accepts the train partition only")
    if type(task_seed) is not int or task_seed < 0:
        raise ValueError("task_seed must be a nonnegative integer")
    if not isinstance(train_x, Tensor):
        raise ValueError("Training input must be a tensor")
    mask = torch.ones(train_x.shape[:3], device=train_x.device, dtype=torch.bool)
    safe = validate_input(train_x, mask).detach()
    if safe.device.type != "cpu":
        raise ValueError("Only CPU calibration is verified")
    rows = safe.permute(0, 2, 3, 1).reshape(-1, 22).double()
    second_moment = rows.T @ rows / len(rows)
    second_moment = (second_moment + second_moment.T) * 0.5
    with scoped_cpu_rng(task_seed + 810001):
        fitter = Tucker2(22, 250, 4, 16, 0.001, 30, 0.01, 64).to(safe)
        history = fitter.fit(safe, mask)
    return {"U": fitter.U.detach().clone(), "V": fitter.V.detach().clone(),
            "second_moment": second_moment,
            "training_trials": torch.tensor(len(safe), dtype=torch.long)}, history


class ZeroCompleter(nn.Module):
    def complete(self, x, mask):
        safe = validate_input(x, mask)
        return x if mask.all().item() else safe

    forward = complete

    def anchor_penalty(self):
        return torch.tensor(0.0)

    def post_step(self):
        pass


class TuckerCompleter(nn.Module):
    """Both arms use the same differentiable float64 observed ridge solve."""

    def __init__(self, u: Tensor, v: Tensor, *, learned: bool = False):
        super().__init__()
        if type(learned) is not bool:
            raise ValueError("learned must be Boolean")
        if (not isinstance(u, Tensor) or not isinstance(v, Tensor) or
                u.shape != (22, 4) or v.shape != (250, 16) or
                u.dtype not in (torch.float32, torch.float64) or u.dtype != v.dtype or
                u.device != v.device or not torch.isfinite(u).all() or not torch.isfinite(v).all()):
            raise ValueError("Expected finite matching U[22,4], V[250,16]")
        for factor in (u, v):
            if not torch.allclose(factor.norm(dim=0), torch.ones_like(factor[0]), rtol=1e-4, atol=1e-5):
                raise ValueError("Tucker factors must have unit columns")
        self.learned = learned
        for name, value in (("U", u), ("V", v)):
            if learned:
                self.register_parameter(name, nn.Parameter(value.detach().clone()))
            else:
                self.register_buffer(name, value.detach().clone())
            self.register_buffer("initial_" + name, value.detach().clone())

    def complete(self, x: Tensor, mask: Tensor) -> Tensor:
        safe = validate_input(x, mask)
        if mask.all().item():
            return x
        core = observed_ridge_core(safe, mask, self.U, self.V, ridge=0.001)
        estimate = torch.einsum("cr,brps,ts->bcpt", self.U, core, self.V)
        if not torch.isfinite(estimate).all().item():
            raise FloatingPointError("Tucker completion produced nonfinite values")
        return torch.where(mask[..., None], x, estimate)

    forward = complete

    def anchor_penalty(self):
        return ((self.U - self.initial_U).square().sum() +
                (self.V - self.initial_V).square().sum()) / (self.U.numel() + self.V.numel())

    @torch.no_grad()
    def post_step(self):
        if self.learned:
            Tucker2._unit_columns(self.U)
            Tucker2._unit_columns(self.V)


class CovarianceCompleter(nn.Module):
    """SPD conditional mean, with identical frozen/learned parameterization."""

    def __init__(self, second_moment: Tensor, *, learned: bool = False):
        super().__init__()
        if type(learned) is not bool:
            raise ValueError("learned must be Boolean")
        if (not isinstance(second_moment, Tensor) or second_moment.shape != (22, 22) or
                not second_moment.is_floating_point() or not torch.isfinite(second_moment).all()):
            raise ValueError("Expected finite training second moment [22,22]")
        moment = second_moment.detach().double()
        if not torch.allclose(moment, moment.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Training second moment must be symmetric")
        lower = torch.linalg.cholesky(moment + 1e-6 * torch.eye(22, device=moment.device, dtype=moment.dtype))
        indices = torch.tril_indices(22, 22, device=moment.device)
        free = lower[indices[0], indices[1]].clone()
        diagonal = indices[0] == indices[1]
        positive = free[diagonal] - 1e-6
        if (positive <= 0).any():
            raise ValueError("Cholesky diagonal must exceed the fixed floor")
        # Stable inverse softplus: z + log(1-exp(-z)).
        free[diagonal] = positive + torch.log(-torch.expm1(-positive))
        self.learned = learned
        if learned:
            self.register_parameter("free", nn.Parameter(free))
        else:
            self.register_buffer("free", free)
        self.register_buffer("initial_free", free.detach().clone())

    def covariance(self):
        if self.free.dtype != torch.float64 or not torch.isfinite(self.free).all():
            raise ValueError("Covariance free parameters must remain finite float64")
        indices = torch.tril_indices(22, 22, device=self.free.device)
        values = torch.where(indices[0] == indices[1], F.softplus(self.free) + 1e-6, self.free)
        lower = self.free.new_zeros(22, 22).index_put((indices[0], indices[1]), values)
        sigma = lower @ lower.T
        if not torch.isfinite(sigma).all():
            raise FloatingPointError("Covariance matrix is nonfinite")
        return sigma

    def complete(self, x: Tensor, mask: Tensor) -> Tensor:
        safe = validate_input(x, mask)
        if mask.all().item():
            return x
        if x.device != self.free.device:
            raise ValueError("Input and covariance parameters must share a device")
        sigma = self.covariance()
        ridge = 0.001 * sigma.diag().mean()
        if not torch.isfinite(ridge).item() or ridge.item() <= 0:
            raise FloatingPointError("Covariance ridge must be finite and positive")
        rows = safe.permute(0, 2, 1, 3).reshape(-1, 22, 250).double()
        flags = mask.permute(0, 2, 1).reshape(-1, 22)
        estimates = torch.zeros_like(rows)
        patterns, inverse = torch.unique(flags, dim=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            known = torch.nonzero(pattern).flatten()
            missing = torch.nonzero(~pattern).flatten()
            if not len(missing):
                continue
            selected = torch.nonzero(inverse == index).flatten()
            system = sigma[known][:, known] + ridge * torch.eye(len(known), device=x.device, dtype=torch.float64)
            beta = torch.linalg.solve(system, sigma[missing][:, known].T)
            prediction = torch.einsum("mo,not->nmt", beta.T, rows[selected][:, known])
            estimates[selected[:, None], missing[None, :]] = prediction
        estimate = estimates.reshape(len(x), 4, 22, 250).permute(0, 2, 1, 3).to(x.dtype)
        if not torch.isfinite(estimate).all().item():
            raise FloatingPointError("Covariance completion produced nonfinite values")
        return torch.where(mask[..., None], x, estimate)

    forward = complete

    def anchor_penalty(self):
        return (self.free - self.initial_free).square().mean()

    def post_step(self):
        self.covariance()  # Fail closed on nonfinite updates; SPD is structural.


def paired_completers(initialization: dict) -> dict[str, nn.Module]:
    """Construct all five strategies from one train-only initialization."""
    if set(initialization) != {"U", "V", "second_moment", "training_trials"}:
        raise ValueError("Initialization has unexpected or missing fields")
    count = initialization["training_trials"]
    if not isinstance(count, Tensor) or count.shape != () or count.dtype != torch.long or count.item() < 1:
        raise ValueError("Initialization requires a positive training trial count")
    models = {"zero": ZeroCompleter()}
    for learned in (False, True):
        suffix = "learned" if learned else "frozen"
        models["covariance_" + suffix] = CovarianceCompleter(initialization["second_moment"], learned=learned)
        models["tucker_" + suffix] = TuckerCompleter(initialization["U"], initialization["V"], learned=learned)
    return {name: models[name] for name in STRATEGY_IDS}


def reconstruction_loss(completed: Tensor, training_target: Tensor, mask: Tensor) -> Tensor:
    """MSE on hidden training samples; an empty target gives differentiable zero.

    A full bypass deliberately has no factor graph. If completed itself needs no
    gradient, the empty loss is a standalone scalar leaf; callers must still skip
    optimizer updates for wholly full batches, including any anchor-only update.
    """
    validate_input(completed, mask)
    validate_input(completed, torch.ones_like(mask))
    validate_input(training_target, torch.ones_like(mask))
    if completed.shape != training_target.shape or completed.device != training_target.device:
        raise ValueError("Reconstruction target must match completed input")
    hidden = ~mask[..., None].expand_as(completed)
    if not hidden.any().item() and not completed.requires_grad:
        return completed.new_zeros((), requires_grad=True)
    # Select before subtracting/squaring; empty selection still preserves autograd.
    return (completed[hidden] - training_target[hidden]).square().sum() / max(int(hidden.sum()), 1)


def completion_state_arrays(model: nn.Module) -> dict[str, np.ndarray]:
    """Primitive state including initial anchors, suitable for an allow_pickle=False NPZ."""
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def restore_completer(strategy: str, arrays: dict[str, np.ndarray]) -> nn.Module:
    """Strict numeric state reconstruction; artifact hashes remain runner-owned."""
    if strategy not in STRATEGY_IDS:
        raise ValueError("Unknown completion strategy")
    if not isinstance(arrays, dict) or any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise ValueError("Completion state must contain primitive numpy arrays")
    expected = set() if strategy == "zero" else (
        {"U", "V", "initial_U", "initial_V"} if strategy.startswith("tucker") else {"free", "initial_free"})
    if set(arrays) != expected:
        raise ValueError("Completion state fields differ from the fixed schema")
    state = {}
    for name, array in arrays.items():
        shape = (22, 4) if name.endswith("U") else (250, 16) if name.endswith("V") else (253,)
        types = (np.dtype("float32"), np.dtype("float64")) if strategy.startswith("tucker") else (np.dtype("float64"),)
        if array.shape != shape or array.dtype not in types or not np.isfinite(array).all():
            raise ValueError(f"Malformed completion state {name}")
        state[name] = torch.from_numpy(array.copy())
    learned = strategy.endswith("learned")
    if strategy == "zero":
        model = ZeroCompleter()
    elif strategy.startswith("tucker"):
        if len({value.dtype for value in state.values()}) != 1:
            raise ValueError("Tucker state dtypes must match")
        model = TuckerCompleter(state["initial_U"], state["initial_V"], learned=learned)
        TuckerCompleter(state["U"], state["V"], learned=learned)  # Unit-column validation.
    else:
        model = CovarianceCompleter(torch.eye(22, dtype=torch.float64), learned=learned)
    model.load_state_dict(state, strict=True)
    if not learned:
        for name in ("U", "V") if strategy.startswith("tucker") else ("free",) if strategy != "zero" else ():
            if not torch.equal(state[name], state["initial_" + name]):
                raise ValueError("Frozen completion state differs from its initialization")
    if isinstance(model, CovarianceCompleter):
        model.covariance()
    return model
