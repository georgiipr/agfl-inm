"""Compact EEGNet classifiers for raw, partially observed EEG trials.

Inputs are ``[batch, channels, windows, window_samples]``. Missing channel
windows are selected out before any convolution, then the EEGNet spatial filter
may mix the remaining observed channels. ``forward_features`` exposes the
retained temporal feature sequence as ``[batch, time, features]`` for controlled
downstream temporal ablations.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class EEGNetClassifier(nn.Module):
    """EEGNet-8,2 style classifier with optional observation-mask conditioning.

    ``mask_conditioned=False`` is the reference arm. When enabled, the original
    Boolean ``[channels, windows]`` availability flags are concatenated to the
    flattened learned sequence immediately before the final classifier.
    """

    def __init__(self, *, channels: int = 22, windows: int = 4,
                 window_samples: int = 250, num_classes: int = 4,
                 temporal_kernel: int = 64, f1: int = 8,
                 depth_multiplier: int = 2, f2: int = 16,
                 pooling: tuple[int, int] = (4, 8),
                 separable_kernel: int = 16, dropout: float = 0.5,
                 mask_conditioned: bool = False,
                 head_type: str = "flatten", temporal_head_width: int = 8,
                 temporal_head_kernel: int = 3, temporal_head_dilation: int = 1,
                 spatial_max_norm: float = 1.0,
                 classifier_max_norm: float = 0.25):
        super().__init__()
        positive = {
            "channels": channels, "windows": windows,
            "window_samples": window_samples, "num_classes": num_classes,
            "temporal_kernel": temporal_kernel, "f1": f1,
            "depth_multiplier": depth_multiplier, "f2": f2,
            "separable_kernel": separable_kernel,
        }
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(pooling, (tuple, list)) or len(pooling) != 2
                or any(type(value) is not int or value < 1 for value in pooling)):
            raise ValueError("pooling must contain two positive integers")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and lie in [0, 1)")
        if type(mask_conditioned) is not bool:
            raise ValueError("mask_conditioned must be boolean")
        if (not math.isfinite(spatial_max_norm) or spatial_max_norm <= 0
                or not math.isfinite(classifier_max_norm) or classifier_max_norm <= 0):
            raise ValueError("weight max-norm constraints must be positive and finite")

        self.channels = channels
        self.windows = windows
        self.window_samples = window_samples
        self.num_classes = num_classes
        self.mask_conditioned = mask_conditioned
        self.spatial_max_norm = float(spatial_max_norm)
        self.classifier_max_norm = float(classifier_max_norm)
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.f2 = f2
        self.temporal_kernel = temporal_kernel
        self.pooling = tuple(pooling)
        self.separable_kernel = separable_kernel
        self.dropout_rate = float(dropout)
        if head_type not in ("flatten", "temporal_conv"):
            raise ValueError("head_type must be 'flatten' or 'temporal_conv'")
        for name, value in (("temporal_head_width", temporal_head_width),
                            ("temporal_head_kernel", temporal_head_kernel),
                            ("temporal_head_dilation", temporal_head_dilation)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.head_type = head_type
        self.temporal_head_width = temporal_head_width
        self.temporal_head_kernel = temporal_head_kernel
        self.temporal_head_dilation = temporal_head_dilation

        expanded = f1 * depth_multiplier
        # EEGNet block 1: temporal filtering, then depthwise spatial filters
        # spanning the full electrode axis BEFORE ELU and temporal pooling.
        self.temporal_conv = nn.Conv2d(
            1, f1, (1, temporal_kernel), padding="same", bias=False)
        self.temporal_bn = nn.BatchNorm2d(f1)
        self.spatial_conv = nn.Conv2d(
            f1, expanded, (channels, 1), groups=f1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(expanded)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, self.pooling[0]))
        self.dropout1 = nn.Dropout(dropout)

        # EEGNet block 2: depthwise temporal convolution and pointwise mixing.
        self.separable_depthwise = nn.Conv2d(
            expanded, expanded, (1, separable_kernel), padding="same",
            groups=expanded, bias=False)
        self.separable_pointwise = nn.Conv2d(expanded, f2, 1, bias=False)
        self.separable_bn = nn.BatchNorm2d(f2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, self.pooling[1]))
        self.dropout2 = nn.Dropout(dropout)

        self.temporal_samples = windows * window_samples
        self.feature_steps = self.temporal_samples // self.pooling[0] // self.pooling[1]
        if self.feature_steps < 1:
            raise ValueError("Pooling leaves no temporal feature positions")
        self.feature_size = f2 * self.feature_steps
        if head_type == "temporal_conv":
            padding = temporal_head_dilation * (temporal_head_kernel - 1) // 2
            self.temporal_head = nn.Conv1d(f2, temporal_head_width, temporal_head_kernel,
                dilation=temporal_head_dilation, padding=padding)
            classifier_inputs = temporal_head_width
        else:
            self.temporal_head = None
            classifier_inputs = self.feature_size
        classifier_inputs += channels * windows if mask_conditioned else 0
        self.classifier = nn.Linear(classifier_inputs, num_classes)
        self.clip_weights()

    def constructor_settings(self) -> dict:
        """Return primitive settings sufficient to reconstruct this module."""
        return {
            "channels": self.channels,
            "windows": self.windows,
            "window_samples": self.window_samples,
            "num_classes": self.num_classes,
            "temporal_kernel": self.temporal_kernel,
            "f1": self.f1,
            "depth_multiplier": self.depth_multiplier,
            "f2": self.f2,
            "pooling": list(self.pooling),
            "separable_kernel": self.separable_kernel,
            "dropout": self.dropout_rate,
            "mask_conditioned": self.mask_conditioned,
            "head_type": self.head_type,
            "temporal_head_width": self.temporal_head_width,
            "temporal_head_kernel": self.temporal_head_kernel,
            "temporal_head_dilation": self.temporal_head_dilation,
            "spatial_max_norm": self.spatial_max_norm,
            "classifier_max_norm": self.classifier_max_norm,
        }

    def _validate_and_select(self, raw: torch.Tensor, mask: torch.Tensor | None):
        expected = (self.channels, self.windows, self.window_samples)
        if not isinstance(raw, torch.Tensor) or raw.ndim != 4 or tuple(raw.shape[1:]) != expected:
            raise ValueError(f"Raw EEG must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
        if not raw.is_floating_point():
            raise ValueError("Raw EEG must have a floating point dtype")
        parameter = self.temporal_conv.weight
        if raw.device != parameter.device:
            raise ValueError("Raw EEG and EEGNet parameters must be on the same device")
        if raw.dtype != parameter.dtype:
            raise ValueError("Raw EEG and EEGNet parameters must have the same dtype")
        if mask is None:
            mask = torch.ones((raw.shape[0], self.channels, self.windows),
                              dtype=torch.bool, device=raw.device)
        if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
            raise ValueError("Availability mask must have boolean dtype")
        if tuple(mask.shape) != (raw.shape[0], self.channels, self.windows):
            raise ValueError("Availability mask must have shape [B,channels,windows]")
        if mask.device != raw.device:
            raise ValueError("Availability mask and raw EEG must be on the same device")
        if not mask.any(dim=1).all():
            raise ValueError("Every trial/window must contain at least one observed channel")

        # torch.where is deliberately the first operation on signal values. In
        # particular, hidden NaNs never enter temporal or spatial mixing.
        selected = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        if not torch.isfinite(selected).all():
            raise ValueError("Observed raw EEG contains non-finite values")
        return selected.reshape(raw.shape[0], self.channels, self.temporal_samples), mask

    def forward_features(self, raw: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``[B, temporal_positions, F2]`` learned temporal features."""
        selected, _ = self._validate_and_select(raw, mask)
        x = selected.unsqueeze(1)  # [B,1,C,T]
        x = self.temporal_bn(self.temporal_conv(x))
        x = self.spatial_conv(x)  # [B,F1*D,1,T], spatial kernel is [C,1]
        x = self.spatial_bn(x)
        x = self.dropout1(self.pool1(self.activation1(x)))
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.separable_bn(x)
        x = self.dropout2(self.pool2(self.activation2(x)))
        # Remove the collapsed electrode axis and return time-major features.
        return x.squeeze(2).transpose(1, 2).contiguous()

    def forward(self, raw: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        sequence = self.forward_features(raw, mask)
        if self.head_type == "temporal_conv":
            features = self.temporal_head(sequence.transpose(1, 2))
            features = self.activation2(features).mean(dim=-1)
        else:
            features = sequence.flatten(start_dim=1)
        if self.mask_conditioned:
            if mask is None:
                mask = torch.ones((raw.shape[0], self.channels, self.windows),
                                  dtype=torch.bool, device=raw.device)
            features = torch.cat((features, mask.reshape(raw.shape[0], -1).to(features.dtype)), dim=1)
        return self.classifier(features)

    @torch.no_grad()
    def clip_weights(self) -> None:
        """Apply EEGNet max-norm constraints to spatial and classifier weights."""
        spatial = self.spatial_conv.weight
        spatial.copy_(torch.renorm(spatial, p=2, dim=0, maxnorm=self.spatial_max_norm))
        classifier = self.classifier.weight
        classifier.copy_(torch.renorm(classifier, p=2, dim=0, maxnorm=self.classifier_max_norm))
