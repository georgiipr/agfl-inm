"""Frozen-backbone routes that feed completed values and retain original flags."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .completion import _validate


def _eegnet_sequence(front_end: nn.Module, raw: Tensor) -> Tensor:
    """Run EEGNet layers directly on already selected/filled raw samples."""
    batch, channels, windows, samples = raw.shape
    x = raw.reshape(batch, channels, windows * samples).unsqueeze(1)
    x = front_end.temporal_bn(front_end.temporal_conv(x))
    x = front_end.spatial_bn(front_end.spatial_conv(x))
    x = front_end.dropout1(front_end.pool1(front_end.activation1(x)))
    x = front_end.separable_depthwise(x)
    x = front_end.separable_pointwise(x)
    x = front_end.separable_bn(x)
    x = front_end.dropout2(front_end.pool2(front_end.activation2(x)))
    return x.squeeze(2).transpose(1, 2).contiguous()


class CompletionAdapter(nn.Module):
    """Evaluate a historical candidate model under zero/covariance/Tucker input.

    The model and all its weights remain frozen. Full masks take the original
    model forward path. Missing masks route filled raw values directly through
    existing layers, while the original Boolean mask remains the flag feature.
    """

    def __init__(self, backbone: nn.Module, strategy: str = "zero", completer: nn.Module | None = None):
        super().__init__()
        if strategy not in ("zero", "covariance", "tucker"):
            raise ValueError("strategy must be zero, covariance, or tucker")
        if strategy == "zero" and completer is not None:
            raise ValueError("zero strategy does not accept a completer")
        if strategy != "zero" and completer is None:
            raise ValueError(f"{strategy} strategy requires a fitted completer")
        if not (hasattr(backbone, "encoder") or hasattr(backbone, "temporal_conv")):
            raise TypeError("Unsupported candidate backbone")
        self.backbone = backbone
        self.strategy = strategy
        self.completer = completer
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        if self.completer is not None:
            self.completer.eval()
            self.completer.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("CompletionAdapter is replay-only and cannot enter training mode")
        return super().train(False)

    def _fill(self, raw: Tensor, mask: Tensor) -> Tensor:
        observed = _validate(raw, mask)
        if self.strategy == "zero":
            return observed
        # Completers validate/select observations again before their solves.
        filled = self.completer.complete(raw, mask)
        if not torch.equal(torch.where(mask.unsqueeze(-1), filled, torch.zeros_like(filled)), observed):
            raise RuntimeError("Completer changed observed samples")
        return filled

    @torch.no_grad()
    def forward(self, raw: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.training or self.backbone.training or (
                self.completer is not None and self.completer.training):
            raise RuntimeError("CompletionAdapter and replay modules must remain in eval mode")
        if not isinstance(raw, Tensor) or raw.ndim != 4 or tuple(raw.shape[1:]) != (22, 4, 250):
            raise ValueError("Replay raw input must have shape [B,22,4,250]")
        if mask is None:
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        _validate(raw, mask)
        if bool(mask.all().item()):
            return self.backbone(raw, mask)
        filled = self._fill(raw, mask)
        flags = mask.reshape(raw.shape[0], -1).to(dtype=raw.dtype)

        if hasattr(self.backbone, "encoder"):
            encoder = self.backbone.encoder
            sequence = _eegnet_sequence(encoder, filled)
            features = sequence.flatten(start_dim=1)
            if encoder.mask_conditioned:
                features = torch.cat((features, flags), dim=1)
            return encoder.classifier(features)

        model = self.backbone
        sequence = _eegnet_sequence(model, filled)
        projected = model.projection(sequence) + model.position_encoding.to(
            device=sequence.device, dtype=sequence.dtype)
        sequence = model.final_norm(model.transformer(projected))
        pooled = sequence.mean(dim=1)
        return model.classifier(torch.cat((pooled, flags), dim=1))
