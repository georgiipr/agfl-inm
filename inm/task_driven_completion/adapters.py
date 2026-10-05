"""Frozen historical Transformer with differentiable completed input."""
from __future__ import annotations

import torch
from torch import nn

from inm.completion_transformer.adapters import _eegnet_sequence
from inm.encoder_candidates.transformer import SpatialTransformer
from .completion import validate_input, ZeroCompleter, TuckerCompleter, CovarianceCompleter
from .protocol import STRATEGY_IDS


class CompletionAdapter(nn.Module):
    """Keep backbone weights/BN/dropout frozen while optimizing completion.

    Unlike the historical replay adapter this forward has no no_grad wrapper.
    Full masks call the original backbone exactly and never touch the completer.
    """

    def __init__(self, backbone: SpatialTransformer, strategy: str = "zero", completer=None):
        super().__init__()
        if not isinstance(backbone, SpatialTransformer):
            raise TypeError("Only the historical SpatialTransformer is supported")
        if strategy not in STRATEGY_IDS:
            raise ValueError("Unknown completion strategy")
        if strategy != "zero" and completer is None:
            raise ValueError("Nonzero strategies require an initialized completer")
        if strategy == "zero" and completer is not None and not isinstance(completer, ZeroCompleter):
            raise ValueError("zero strategy accepts only ZeroCompleter")
        if strategy != "zero":
            expected = TuckerCompleter if strategy.startswith("tucker") else CovarianceCompleter
            if not isinstance(completer, expected) or completer.learned != strategy.endswith("learned"):
                raise ValueError("Strategy must match completer family and learned/frozen status")
        self.backbone = backbone
        self.strategy = strategy
        self.completer = completer if completer is not None else ZeroCompleter()
        self.backbone.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, raw, mask=None):
        if any(module.training for module in self.backbone.modules()):
            raise RuntimeError("Frozen backbone must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("Frozen backbone parameters must not require gradients")
        if mask is None:
            if not isinstance(raw, torch.Tensor):
                raise ValueError("Raw input must be a tensor")
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        observed = validate_input(raw, mask)
        parameter = self.backbone.temporal_conv.weight
        if raw.device != parameter.device or raw.dtype != parameter.dtype:
            raise ValueError("Raw input and backbone must share dtype and device")
        if mask.all().item() or self.strategy == "zero":
            # Exact historical path for the zero control as well as full input.
            return self.backbone(raw, mask)
        filled = self.completer.complete(raw, mask)
        if not torch.equal(torch.where(mask[..., None], filled, torch.zeros_like(filled)), observed):
            raise RuntimeError("Completer changed observed samples")
        if not torch.isfinite(filled).all():
            raise FloatingPointError("Completed input must be finite")
        model = self.backbone
        sequence = _eegnet_sequence(model, filled)
        projected = model.projection(sequence) + model.position_encoding.to(sequence)
        encoded = model.final_norm(model.transformer(projected))
        flags = mask.reshape(len(raw), -1).to(raw.dtype)
        return model.classifier(torch.cat((encoded.mean(dim=1), flags), dim=1))
