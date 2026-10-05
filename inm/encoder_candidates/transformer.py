"""Compact temporal Transformer readout over a spatial EEGNet front end."""
from __future__ import annotations

import math

import torch
from torch import nn

from inm.baselines.eegnet import EEGNetClassifier


def sinusoidal_position_encoding(length: int, d_model: int, *, device=None,
                                 dtype=None) -> torch.Tensor:
    """Return fixed ``[1,length,d_model]`` sin/cos temporal positions."""
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if type(d_model) is not int or d_model < 2 or d_model % 2:
        raise ValueError("d_model must be a positive even integer")
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.empty(length, d_model, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    if dtype is not None:
        encoding = encoding.to(dtype=dtype)
    return encoding.unsqueeze(0)


class SpatialTransformer(nn.Module):
    """EEGNet spatial activations followed by one compact temporal layer.

    Inputs are normalized raw EEG ``[B,22,4,250]`` and an optional Boolean
    availability mask ``[B,22,4]``. Missing samples are selected to zero before
    any convolution or spatial mixing.
    """

    D_MODEL = 32
    HEADS = 4
    FEEDFORWARD = 64
    DROPOUT = 0.1
    FEATURE_STEPS = 31

    def __init__(self, *, seed: int = 0):
        super().__init__()
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.channels, self.windows, self.window_samples = 22, 4, 250
        self.temporal_samples = self.windows * self.window_samples
        # Construct the exact reference EEGNet initialization in an isolated
        # RNG scope, then retain only the layers used to form temporal features.
        # This keeps front-end tensors identical to SpatialEEGNet(seed),
        # regardless of how many Transformer-head parameters are added below.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            front_end = EEGNetClassifier(**self._fixed_encoder_settings())
        for name in ("temporal_conv", "temporal_bn", "spatial_conv", "spatial_bn",
                     "activation1", "pool1", "dropout1", "separable_depthwise",
                     "separable_pointwise", "separable_bn", "activation2",
                     "pool2", "dropout2"):
            setattr(self, name, getattr(front_end, name))
        self.spatial_max_norm = front_end.spatial_max_norm
        del front_end

        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(self.FEATURE_STEPS, self.D_MODEL),
            persistent=True,
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.projection = nn.Linear(16, self.D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=self.D_MODEL,
                nhead=self.HEADS,
                dim_feedforward=self.FEEDFORWARD,
                dropout=self.DROPOUT,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=1, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(self.D_MODEL)
            self.classifier = nn.Linear(self.D_MODEL + 22 * 4, 4)
        self.clip_weights()

    @staticmethod
    def _fixed_encoder_settings():
        return {"channels": 22, "windows": 4, "window_samples": 250,
                "num_classes": 4, "temporal_kernel": 64, "f1": 8,
                "depth_multiplier": 2, "f2": 16, "pooling": [4, 8],
                "separable_kernel": 16, "dropout": 0.5,
                "mask_conditioned": True, "head_type": "flatten"}

    @staticmethod
    def _fixed_transformer_settings():
        return {"d_model": 32, "heads": 4, "layers": 1,
                "dim_feedforward": 64, "activation": "gelu", "dropout": 0.1,
                "norm_first": True, "batch_first": True,
                "position_encoding": "fixed_sinusoidal", "pooling": "mean"}

    def constructor_settings(self):
        return {"seed": self.seed, "encoder": self._fixed_encoder_settings(),
                "transformer": self._fixed_transformer_settings(),
                "num_classes": 4, "mask_flags": 88}

    def _select(self, raw, mask):
        expected = (22, 4, 250)
        if not isinstance(raw, torch.Tensor) or raw.ndim != 4 or tuple(raw.shape[1:]) != expected:
            raise ValueError("Raw EEG must have shape [B,22,4,250]")
        if not raw.is_floating_point():
            raise ValueError("Raw EEG must have a floating point dtype")
        parameter = self.temporal_conv.weight
        if raw.device != parameter.device:
            raise ValueError("Raw EEG and model parameters must be on the same device")
        if raw.dtype != parameter.dtype:
            raise ValueError("Raw EEG and model parameters must have the same dtype")
        if mask is None:
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        if (not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool
                or tuple(mask.shape) != tuple(raw.shape[:3]) or mask.device != raw.device):
            raise ValueError("Availability mask must be boolean [B,22,4] on the input device")
        if not mask.any(dim=1).all():
            raise ValueError("Every trial/window must contain at least one observed channel")
        # This must precede every arithmetic/filtering operation on raw values.
        selected = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        if not torch.isfinite(selected).all():
            raise ValueError("Observed raw EEG contains non-finite values")
        return selected, mask

    def _validate_and_select(self, raw, mask):
        selected, mask = self._select(raw, mask)
        return selected.reshape(raw.shape[0], self.channels, self.temporal_samples), mask

    def _front_end_features(self, raw, mask=None):
        return EEGNetClassifier.forward_features(self, raw, mask), self._normalize_mask(raw, mask)

    def _normalize_mask(self, raw, mask):
        if mask is None:
            return torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        return mask

    def forward_features(self, raw, mask=None):
        """Return Transformer sequence features ``[B,31,32]``."""
        sequence, _ = self._front_end_features(raw, mask)
        projected = self.projection(sequence) + self.position_encoding.to(
            device=sequence.device, dtype=sequence.dtype)
        return self.final_norm(self.transformer(projected))

    def forward(self, raw, mask=None):
        sequence = self.forward_features(raw, mask)
        if mask is None:
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        pooled = sequence.mean(dim=1)
        availability = mask.reshape(raw.shape[0], -1).to(dtype=pooled.dtype)
        return self.classifier(torch.cat((pooled, availability), dim=1))

    @torch.no_grad()
    def clip_weights(self):
        spatial = self.spatial_conv.weight
        spatial.copy_(torch.renorm(spatial, p=2, dim=0,
                                   maxnorm=self.spatial_max_norm))
