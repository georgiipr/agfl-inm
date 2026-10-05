"""Frozen train-only raw-signal completion controls for replay adapters."""
from __future__ import annotations

from contextlib import contextmanager
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


def _validate(x: Tensor, mask: Tensor, channels: int = 22, windows: int = 4):
    if not isinstance(x, Tensor) or x.ndim != 4 or tuple(x.shape[1:3]) != (channels, windows):
        raise ValueError(f"Raw input must have shape [B,{channels},{windows},T]")
    if not x.is_floating_point():
        raise TypeError("Raw input must be floating point")
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool or tuple(mask.shape) != tuple(x.shape[:3]):
        raise ValueError("Availability mask must be boolean [B,C,P]")
    if mask.device != x.device:
        raise ValueError("Input and availability mask must share a device")
    if (mask.sum(dim=1) == 0).any().item():
        raise ValueError("Every window must contain at least one observed channel")
    observed = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
    if not torch.isfinite(observed).all().item():
        raise ValueError("Observed input samples must be finite")
    return observed


def preserve_observed(x: Tensor, estimate: Tensor, mask: Tensor) -> Tensor:
    """Use estimates only for missing channel-windows; retain observed samples exactly."""
    if estimate.shape != x.shape or estimate.device != x.device:
        raise ValueError("Estimate must match input shape and device")
    if not torch.isfinite(estimate).all().item():
        raise FloatingPointError("Completion estimate is non-finite")
    return torch.where(mask.unsqueeze(-1), x, estimate)


class CovarianceCompleter(nn.Module):
    """Fixed conditional channel-second-moment completion."""

    def __init__(self, channels: int = 22, ridge_scale: float = 0.001):
        super().__init__()
        if type(channels) is not int or channels < 1 or not np.isfinite(ridge_scale) or ridge_scale <= 0:
            raise ValueError("channels and ridge_scale must be positive")
        self.channels, self.ridge_scale = channels, float(ridge_scale)
        self.register_buffer("second_moment", torch.zeros(channels, channels))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("training_samples", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def fit(self, train_x: Tensor) -> "CovarianceCompleter":
        if not isinstance(train_x, Tensor) or train_x.ndim != 4 or train_x.shape[1] != self.channels:
            raise ValueError(f"Training raw input must have shape [N,{self.channels},P,T]")
        if not train_x.is_floating_point() or not torch.isfinite(train_x).all().item():
            raise ValueError("Training raw input must be finite and floating point")
        rows = train_x.permute(0, 2, 3, 1).reshape(-1, self.channels).to(torch.float64)
        covariance = rows.T @ rows / rows.shape[0]
        covariance = (covariance + covariance.T) * 0.5
        if not torch.isfinite(covariance).all().item() or (covariance.diag() <= 0).any().item():
            raise ValueError("Training channel second moments must be finite and positive")
        self.second_moment.copy_(covariance.to(self.second_moment))
        self.training_samples.fill_(rows.shape[0])
        self.fitted.fill_(True)
        return self

    @torch.no_grad()
    def complete(self, x: Tensor, mask: Tensor) -> Tensor:
        observed = _validate(x, mask, self.channels, x.shape[2])
        if not self.fitted.item():
            raise RuntimeError("Fit the covariance completer on training data or load its state")
        cov = self.second_moment.to(dtype=x.dtype, device=x.device)
        ridge = self.ridge_scale * cov.diag().mean()
        # Each channel/window has its own observed set. The small solve is
        # grouped by mask so repeated availability patterns share one factorization.
        flat_x = observed.permute(0, 2, 1, 3).reshape(-1, self.channels, x.shape[-1])
        flat_mask = mask.permute(0, 2, 1).reshape(-1, self.channels)
        estimates = torch.zeros_like(flat_x)
        patterns, inverse = torch.unique(flat_mask, dim=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            missing = torch.nonzero(~pattern, as_tuple=False).flatten()
            known = torch.nonzero(pattern, as_tuple=False).flatten()
            if not len(missing):
                continue
            rows = torch.nonzero(inverse == index, as_tuple=False).flatten()
            block = cov[known][:, known]
            cross = cov[missing][:, known]
            # Solve (Sigma_oo + ridge I) beta = Sigma_om.T; then x_m=beta.T x_o.
            beta = torch.linalg.solve(block + ridge * torch.eye(len(known), device=x.device, dtype=x.dtype), cross.T)
            estimates[rows[:, None], missing[None, :], :] = torch.einsum(
                "mo,not->nmt", beta.T, flat_x[rows][:, known, :])
        estimate = estimates.reshape(x.shape[0], x.shape[2], self.channels, x.shape[3]).permute(0, 2, 1, 3)
        return preserve_observed(x, estimate, mask)


@contextmanager
def _scoped_cpu_rng(seed: int):
    """Seed Python, NumPy and CPU Torch for a fit, restoring ambient states."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class TuckerCompleter(nn.Module):
    """Train-only frozen Tucker-2 completion, using the existing implementation."""

    def __init__(self, *, channels: int = 22, windows: int = 4, samples: int = 250,
                 rank_channels: int = 4, rank_features: int = 16, ridge: float = 0.001,
                 epochs: int = 30, learning_rate: float = 0.01, batch_size: int = 64,
                 seed_offset: int = 810001):
        super().__init__()
        if windows < 1 or samples < 1:
            raise ValueError("windows and samples must be positive")
        self.channels, self.windows, self.samples = channels, windows, samples
        self.rank_channels, self.rank_features = rank_channels, rank_features
        self.ridge, self.epochs = float(ridge), epochs
        self.learning_rate, self.batch_size = float(learning_rate), batch_size
        self.seed_offset = seed_offset
        from inm.tensor_attention import Tucker2
        self.model = Tucker2(channels, samples, rank_channels, rank_features, ridge,
                             epochs, learning_rate, batch_size)

    def fit(self, train_x: Tensor, task_seed: int) -> list[dict[str, float | int]]:
        _validate(train_x, torch.ones(train_x.shape[:3], dtype=torch.bool, device=train_x.device),
                  self.channels, self.windows)
        if train_x.shape[-1] != self.samples or train_x.device.type != "cpu":
            raise ValueError("Tucker fitting expects configured raw length on CPU")
        with _scoped_cpu_rng(task_seed + self.seed_offset):
            history = self.model.fit(
                train_x, torch.ones(train_x.shape[:3], dtype=torch.bool, device=train_x.device),
                epochs=self.epochs, lr=self.learning_rate, batch_size=self.batch_size)
        return history

    @torch.no_grad()
    def complete(self, x: Tensor, mask: Tensor) -> Tensor:
        if x.shape[-1] != self.samples or x.shape[2] != self.windows:
            raise ValueError("Input dimensions do not match Tucker factors")
        return self.model.complete(x, mask)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.complete(x, mask)


def completion_state_arrays(covariance: CovarianceCompleter, tucker: TuckerCompleter) -> dict[str, np.ndarray]:
    """Return the complete, primitive-only state stored with a completed task."""
    arrays: dict[str, np.ndarray] = {}
    for name, value in covariance.state_dict().items():
        arrays[f"covariance__{name}"] = value.detach().cpu().numpy().copy()
    for name, value in tucker.model.state_dict().items():
        arrays[f"tucker__{name}"] = value.detach().cpu().numpy().copy()
    settings = {
        "channels": covariance.channels,
        "ridge_scale": covariance.ridge_scale,
        "windows": tucker.windows,
        "samples": tucker.samples,
        "rank_channels": tucker.rank_channels,
        "rank_features": tucker.rank_features,
        "ridge": tucker.ridge,
        "epochs": tucker.epochs,
        "learning_rate": tucker.learning_rate,
        "batch_size": tucker.batch_size,
        "seed_offset": tucker.seed_offset,
    }
    for name, value in settings.items():
        arrays[f"setting__{name}"] = np.asarray(value)
    return arrays


def _validate_history(history, expected_epochs: int) -> None:
    if type(expected_epochs) is not int or expected_epochs < 1:
        raise ValueError("expected Tucker epoch count must be a positive integer")
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise ValueError("completion history length differs from fitted Tucker epoch count")
    required = {"epoch", "objective", "reconstruction_mse", "core_penalty"}
    for index, row in enumerate(history, 1):
        if (not isinstance(row, dict) or set(row) != required or
                type(row.get("epoch")) is not int or row["epoch"] != index):
            raise ValueError("completion history epoch sequence or fields are malformed")
        for key in required - {"epoch"}:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"completion history {key} must be finite and nonnegative")


def load_completion_state(path: str | Path, history, *, expected_epochs: int,
                          expected_settings: dict | None = None) -> tuple[CovarianceCompleter, TuckerCompleter]:
    """Safely load and numerically validate a completed task's primitive state."""
    _validate_history(history, expected_epochs)
    with np.load(path, allow_pickle=False) as data:
        settings_names = ("channels", "ridge_scale", "windows", "samples", "rank_channels",
                          "rank_features", "ridge", "epochs", "learning_rate", "batch_size", "seed_offset")
        if any(f"setting__{name}" not in data.files for name in settings_names):
            raise ValueError("completion state omits required settings")
        settings = {}
        integer_settings = {"channels", "windows", "samples", "rank_channels", "rank_features", "epochs", "batch_size", "seed_offset"}
        for name in settings_names:
            value = data[f"setting__{name}"]
            if value.shape != ():
                raise ValueError(f"completion state setting {name} must be scalar")
            scalar = value.item()
            if name in integer_settings:
                if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, np.integer)):
                    raise ValueError(f"completion state setting {name} must be an integer")
            elif isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (float, int, np.floating, np.integer)) or not np.isfinite(scalar):
                raise ValueError(f"completion state setting {name} must be finite numeric")
            settings[name] = scalar
        if settings["epochs"] != expected_epochs:
            raise ValueError("completion state epoch setting differs from task history")
        if expected_settings is not None:
            for name, expected in expected_settings.items():
                if name not in settings or settings[name] != expected:
                    raise ValueError(f"completion state setting {name} differs from the declared study")
        covariance = CovarianceCompleter(int(settings["channels"]), float(settings["ridge_scale"]))
        tucker = TuckerCompleter(channels=int(settings["channels"]), windows=int(settings["windows"]),
            samples=int(settings["samples"]), rank_channels=int(settings["rank_channels"]),
            rank_features=int(settings["rank_features"]), ridge=float(settings["ridge"]),
            epochs=int(settings["epochs"]), learning_rate=float(settings["learning_rate"]),
            batch_size=int(settings["batch_size"]), seed_offset=int(settings["seed_offset"]))
        expected = ({f"covariance__{name}" for name in covariance.state_dict()} |
                    {f"tucker__{name}" for name in tucker.model.state_dict()} |
                    {f"setting__{name}" for name in settings_names})
        if set(data.files) != expected:
            raise ValueError("completion state arrays differ from the declared schema")
        cov_state, tuck_state = {}, {}
        for prefix, module, target in (("covariance__", covariance, cov_state),
                                       ("tucker__", tucker.model, tuck_state)):
            for name, current in module.state_dict().items():
                array = data[prefix + name]
                if tuple(array.shape) != tuple(current.shape):
                    raise ValueError(f"completion state {prefix + name} has invalid dimensions")
                expected_dtype = current.numpy().dtype
                if array.dtype != expected_dtype:
                    raise ValueError(f"completion state {prefix + name} has invalid type")
                if current.is_floating_point() and not np.isfinite(array).all():
                    raise ValueError(f"completion state {prefix + name} contains non-finite values")
                target[name] = torch.as_tensor(np.array(array, copy=True), dtype=current.dtype)
        covariance.load_state_dict(cov_state, strict=True)
        tucker.model.load_state_dict(tuck_state, strict=True)

    if not covariance.fitted.item() or int(covariance.training_samples.item()) < 1:
        raise ValueError("covariance completion state is not marked fitted")
    moment = covariance.second_moment
    if (not torch.isfinite(moment).all().item() or (moment.diag() <= 0).any().item() or
            not torch.allclose(moment, moment.T, rtol=1e-6, atol=1e-7)):
        raise ValueError("covariance second moment is non-finite, asymmetric, or degenerate")
    model = tucker.model
    if not model.fitted.item() or int(model.fit_steps.item()) != expected_epochs:
        raise ValueError("Tucker state fitted flag or epoch count differs from history")
    if (not torch.isfinite(model.U).all().item() or not torch.isfinite(model.V).all().item() or
            not torch.allclose(torch.linalg.vector_norm(model.U, dim=0), torch.ones(model.rank_channels), rtol=1e-4, atol=1e-5) or
            not torch.allclose(torch.linalg.vector_norm(model.V, dim=0), torch.ones(model.rank_features), rtol=1e-4, atol=1e-5)):
        raise ValueError("Tucker factors are non-finite or not unit-normalized")
    if model.training_seen_counts.shape != (model.channels,) or (model.training_seen_counts < 1).any().item():
        raise ValueError("Tucker training-seen counts are malformed")
    covariance.eval()
    tucker.eval()
    return covariance, tucker
