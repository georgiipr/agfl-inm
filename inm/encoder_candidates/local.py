"""Matched local encoder variants for the encoder-candidate study."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from inm.model import EEGWindowEncoder, FeatureClassifier


class LocalPowerEncoder(nn.Module):
    """Window-local multiscale log-variance features from raw EEG."""

    def __init__(self, *, channels=22, windows=4, window_samples=250,
                 kernels=(16, 32, 64, 128), filters_per_branch=8,
                 input_channels=1, stride=1, bias=False, log_epsilon=1e-6,
                 padding="explicit_zero_same", variance="population_over_250_responses",
                 layout="B22W4F32"):
        super().__init__()
        if (type(channels) is not int or channels < 1 or type(windows) is not int
                or windows < 1 or type(window_samples) is not int or window_samples < 1):
            raise ValueError("channels, windows and window_samples must be positive integers")
        kernels = tuple(kernels)
        if not kernels or any(type(k) is not int or k < 1 for k in kernels):
            raise ValueError("kernels must be positive integers")
        if type(filters_per_branch) is not int or filters_per_branch < 1:
            raise ValueError("filters_per_branch must be a positive integer")
        if input_channels != 1 or stride != 1 or bias is not False:
            raise ValueError("Local power branches require one input, stride 1 and no bias")
        if padding != "explicit_zero_same" or variance != "population_over_250_responses":
            raise ValueError("Unsupported local power padding or variance convention")
        if not isinstance(log_epsilon, (float, int)) or log_epsilon <= 0:
            raise ValueError("log_epsilon must be positive")
        features = len(kernels) * filters_per_branch
        if layout != f"B{channels}W{windows}F{features}":
            raise ValueError("layout must describe the configured output dimensions")
        self.channels, self.windows, self.window_samples = channels, windows, window_samples
        self.kernels = kernels
        self.filters_per_branch = filters_per_branch
        self.input_channels, self.stride, self.bias = input_channels, stride, bias
        self.log_epsilon = float(log_epsilon)
        self.padding, self.variance, self.layout = padding, variance, layout
        self.features = features
        self.branches = nn.ModuleList([
            nn.Conv1d(input_channels, filters_per_branch, kernel_size=kernel,
                      stride=stride, padding=0, bias=bias)
            for kernel in kernels
        ])

    def constructor_settings(self):
        return {"channels": self.channels, "windows": self.windows,
                "window_samples": self.window_samples, "kernels": list(self.kernels),
                "filters_per_branch": self.filters_per_branch,
                "input_channels": self.input_channels, "stride": self.stride,
                "bias": self.bias, "log_epsilon": self.log_epsilon,
                "padding": self.padding, "variance": self.variance,
                "layout": self.layout}

    def forward_features(self, raw, mask=None):
        if raw.ndim != 4 or tuple(raw.shape[1:]) != (
                self.channels, self.windows, self.window_samples):
            raise ValueError(f"Raw EEG must have shape [B,{self.channels},{self.windows},{self.window_samples}]")
        batch = raw.shape[0]
        mask = _availability(mask, batch, self.channels, self.windows, raw.device)
        observed = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        if not torch.isfinite(observed).all():
            raise ValueError("Observed raw EEG contains non-finite values")
        local = observed.reshape(batch * self.channels * self.windows,
                                 self.input_channels, self.window_samples)
        features = []
        for kernel, branch in zip(self.kernels, self.branches):
            left, right = (kernel - 1) // 2, kernel // 2
            response = branch(F.pad(local, (left, right), mode="constant", value=0.0))
            # Population variance across all temporal responses, independently
            # for each example/channel/window/filter.
            power = torch.log(response.var(dim=-1, unbiased=False) + self.log_epsilon)
            features.append(power)
        encoded = torch.cat(features, dim=-1).reshape(
            batch, self.channels, self.windows, self.features)
        return torch.where(mask[..., None], encoded, torch.zeros_like(encoded))

    def forward(self, raw, mask=None):
        return self.forward_features(raw, mask)

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self

    def clip_weights(self):
        """No max-norm constraint is part of the local power design."""
        return None


def _availability(mask, batch, channels, windows, device):
    if mask is None:
        return torch.ones(batch, channels, windows, device=device, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("Availability must be boolean [batch, channels, windows]")
    if tuple(mask.shape) != (batch, channels, windows):
        raise ValueError("Availability must be boolean [batch, channels, windows]")
    if mask.device != device:
        raise ValueError("Availability and input must be on the same device")
    if not mask.any(dim=1).all():
        raise ValueError("Every trial/window must contain at least one observed channel")
    return mask


class LocalClassifier(nn.Module):
    """One local encoder paired with the fixed shared spatial-attention head."""

    def __init__(self, kind, *, seed=0):
        super().__init__()
        if kind not in ("local_control", "local_power"):
            raise ValueError("kind must be 'local_control' or 'local_power'")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.kind, self.seed = kind, seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = (EEGWindowEncoder() if kind == "local_control" else LocalPowerEncoder())
        # Reset RNG immediately before constructing the shared head. Encoder
        # parameter count and initialization draws therefore cannot affect it.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.head = FeatureClassifier("mha", representation="baseline", channels=22,
                windows=4, features=32, num_classes=4, dim=32, heads=4, depth=1,
                dropout=0.1)
        self._features_frozen = False
        self.clip_weights()

    def constructor_settings(self):
        return {"kind": self.kind, "seed": self.seed,
                "encoder": ({"f1": 16, "d": 2, "f2": 32, "kernel_length": 32,
                             "pool1": 8, "pool2": 16, "dropout": 0.5}
                            if self.kind == "local_control"
                            else self.encoder.constructor_settings()),
                "head": {"attention": "MHA", "representation": "baseline",
                         "channels": 22, "windows": 4, "features": 32,
                         "num_classes": 4, "dim": 32, "heads": 4,
                         "depth": 1, "dropout": 0.1}}

    def forward_features(self, raw, mask=None):
        if raw.ndim != 4 or tuple(raw.shape[1:]) != (22, 4, 250):
            raise ValueError("Raw EEG must have shape [B,22,4,250]")
        availability = _availability(mask, raw.shape[0], 22, 4, raw.device)
        observed = torch.where(availability[..., None], raw, torch.zeros_like(raw))
        if not torch.isfinite(observed).all():
            raise ValueError("Observed raw EEG contains non-finite values")
        if self.kind == "local_control":
            flattened = observed.reshape(raw.shape[0], 22, 1000)
            return self.encoder(flattened, availability)
        return self.encoder.forward_features(observed, availability)

    def forward(self, raw, mask=None):
        features = self.forward_features(raw, mask)
        return self.head(features, mask)

    def freeze_features(self):
        self._features_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        return self

    def unfreeze_features(self):
        """Explicitly re-enable feature learning for a new training operation."""
        self._features_frozen = False
        self.encoder.requires_grad_(True)
        self.encoder.train(self.training)
        if self.kind == "local_control":
            self.encoder.frozen = False
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._features_frozen:
            self.encoder.eval()
        return self

    def clip_weights(self):
        if self.kind == "local_control":
            self.encoder.clip_weights()
        self.head.clip_weights()
