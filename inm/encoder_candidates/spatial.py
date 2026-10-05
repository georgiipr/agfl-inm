"""Spatial EEGNet and fixed-FIR spatial filter-bank candidates."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.signal import firwin
from torch import nn
from torch.nn import functional as F

from inm.baselines.eegnet import EEGNetClassifier


class SpatialEEGNet(nn.Module):
    """Mask-conditioned EEGNet reference for raw ``[B,22,4,250]`` inputs."""

    def __init__(self, *, seed: int = 0):
        super().__init__()
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        settings = self._fixed_encoder_settings()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = EEGNetClassifier(**settings)

    @staticmethod
    def _fixed_encoder_settings():
        return {"channels": 22, "windows": 4, "window_samples": 250,
                "num_classes": 4, "temporal_kernel": 64, "f1": 8,
                "depth_multiplier": 2, "f2": 16, "pooling": [4, 8],
                "separable_kernel": 16, "dropout": 0.5,
                "mask_conditioned": True, "head_type": "flatten"}

    def constructor_settings(self):
        return {"seed": self.seed, "encoder": self._fixed_encoder_settings()}

    def forward_features(self, raw, mask=None):
        return self.encoder.forward_features(raw, mask)

    def forward(self, raw, mask=None):
        return self.encoder(raw, mask)

    def clip_weights(self):
        self.encoder.clip_weights()


class SpatialFilterBank(nn.Module):
    """Window-local fixed FIR bands followed by learned spatial log-variance.

    The learned outputs are latent spatial components. Their index is not an
    original electrode identity; the original channel/window availability
    flags are separately retained at the classifier input.
    """

    BANDS_HZ = ((4.0, 8.0), (8.0, 12.0), (12.0, 16.0),
                (16.0, 20.0), (20.0, 24.0), (24.0, 30.0))

    def __init__(self, *, seed: int = 0, channels: int = 22,
                 windows: int = 4, window_samples: int = 250,
                 bands_hz=((4, 8), (8, 12), (12, 16),
                           (16, 20), (20, 24), (24, 30)),
                 fir_taps: int = 81, fir_window: str = "hamming",
                 sampling_rate: int = 250, spatial_filters_per_band: int = 4,
                 spatial_filter_max_norm: float = 1.0,
                 log_epsilon: float = 1e-6, dropout: float = 0.5,
                 num_classes: int = 4):
        super().__init__()
        bands = tuple(tuple(float(edge) for edge in band) for band in bands_hz)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if (channels, windows, window_samples, fir_taps, sampling_rate,
                spatial_filters_per_band, num_classes) != (22, 4, 250, 81, 250, 4, 4):
            raise ValueError("SpatialFilterBank dimensions and fixed FIR settings are part of the protocol")
        if bands != self.BANDS_HZ:
            raise ValueError("bands_hz must equal the six fixed protocol bands")
        if fir_window != "hamming" or type(fir_window) is not str:
            raise ValueError("fir_window must be 'hamming'")
        if (not math.isfinite(spatial_filter_max_norm) or spatial_filter_max_norm <= 0
                or not math.isfinite(log_epsilon) or log_epsilon <= 0
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("max norm and epsilon must be positive; dropout must lie in [0,1)")
        coeffs = np.stack([
            firwin(fir_taps, band, pass_zero=False, fs=sampling_rate,
                   window=fir_window).astype(np.float32)
            for band in bands
        ])
        self.register_buffer("fir_coefficients", torch.from_numpy(coeffs), persistent=True)
        self.seed, self.channels, self.windows = seed, channels, windows
        self.window_samples, self.bands_hz = window_samples, bands
        self.fir_taps, self.fir_window = fir_taps, fir_window
        self.sampling_rate = sampling_rate
        self.spatial_filters_per_band = spatial_filters_per_band
        self.spatial_filter_max_norm = float(spatial_filter_max_norm)
        self.log_epsilon, self.dropout_rate = float(log_epsilon), float(dropout)
        self.num_classes = num_classes
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.spatial_filters = nn.Parameter(torch.empty(
                len(bands), spatial_filters_per_band, channels))
            nn.init.normal_(self.spatial_filters, mean=0.0, std=0.05)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(windows * len(bands) * spatial_filters_per_band
                                        + channels * windows, num_classes)
        self.clip_weights()

    def constructor_settings(self):
        return {"seed": self.seed, "channels": self.channels,
                "windows": self.windows, "window_samples": self.window_samples,
                "bands_hz": [list(band) for band in self.bands_hz],
                "fir_taps": self.fir_taps, "fir_window": self.fir_window,
                "sampling_rate": self.sampling_rate,
                "spatial_filters_per_band": self.spatial_filters_per_band,
                "spatial_filter_max_norm": self.spatial_filter_max_norm,
                "log_epsilon": self.log_epsilon, "dropout": self.dropout_rate,
                "num_classes": self.num_classes}

    def _select(self, raw, mask):
        if (not isinstance(raw, torch.Tensor) or raw.ndim != 4
                or tuple(raw.shape[1:]) != (self.channels, self.windows, self.window_samples)):
            raise ValueError("Raw EEG must have shape [B,22,4,250]")
        if not raw.is_floating_point():
            raise ValueError("Raw EEG must have a floating point dtype")
        if raw.device != self.spatial_filters.device or raw.dtype != self.spatial_filters.dtype:
            raise ValueError("Raw EEG and model parameters must have the same device and dtype")
        if mask is None:
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        if (not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool
                or tuple(mask.shape) != tuple(raw.shape[:3]) or mask.device != raw.device):
            raise ValueError("Availability must be boolean [B,22,4] on the input device")
        if not mask.any(dim=1).all():
            raise ValueError("Every trial/window must contain at least one observed channel")
        selected = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        if not torch.isfinite(selected).all():
            raise ValueError("Observed raw EEG contains non-finite values")
        return selected, mask

    def forward_features(self, raw, mask=None):
        """Return ordered ``[B,4,6,4]`` window/band/latent-component features."""
        selected, mask = self._select(raw, mask)
        batch = selected.shape[0]
        # Each channel/window is a separate length-250 sequence. Padding is
        # local to that sequence, so no FIR response spans adjacent windows.
        signals = selected.reshape(batch * self.channels * self.windows, 1,
                                   self.window_samples)
        pad = self.fir_taps // 2
        band_responses = []
        for coefficients in self.fir_coefficients:
            response = F.conv1d(F.pad(signals, (pad, pad)),
                                coefficients.reshape(1, 1, -1))
            band_responses.append(response.reshape(
                batch, self.channels, self.windows, self.window_samples))
        filtered = torch.stack(band_responses, dim=3)  # [B,C,W,K,T]
        # Learned spatial filters combine the original electrode axis into
        # latent components independently for each window and fixed band.
        components = torch.einsum("bcwkt,koc->bwkot", filtered, self.spatial_filters)
        features = torch.log(components.var(dim=-1, unbiased=False) + self.log_epsilon)
        return features  # [B,W,K,O]

    def forward(self, raw, mask=None):
        features = self.forward_features(raw, mask)
        if mask is None:
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        combined = torch.cat((features.flatten(start_dim=1),
                              mask.reshape(raw.shape[0], -1).to(features.dtype)), dim=1)
        return self.classifier(self.dropout(combined))

    @torch.no_grad()
    def clip_weights(self):
        norms = self.spatial_filters.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        self.spatial_filters.mul_((self.spatial_filter_max_norm / norms).clamp(max=1.0))
