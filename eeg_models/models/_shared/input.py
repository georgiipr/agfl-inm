"""Explicit signal availability, with optional training-fitted completion."""
import torch
from torch import nn


class SignalInput(nn.Module):
    def __init__(self, metadata, window_samples):
        super().__init__()
        self.channels, self.samples = int(metadata['channels']), int(metadata['samples'])
        if type(window_samples) is not int or window_samples < 1 or self.samples % window_samples:
            raise ValueError('window_samples must divide the full trial length')
        self.window_samples = window_samples
        self.windows = self.samples // window_samples
        self.completion = None

    def forward(self, raw, mask=None):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (self.channels, self.samples):
            raise ValueError(f'Expected EEG [B,{self.channels},{self.samples}]')
        if mask is None:
            if not bool(torch.isfinite(raw).all()):
                raise ValueError('Observed EEG contains nonfinite samples')
            return raw, None
        if (mask.dtype != torch.bool or mask.device != raw.device
                or tuple(mask.shape) != (raw.shape[0], self.channels, self.windows)):
            raise ValueError('Availability must be boolean [B,C,P] on the signal device')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Every window must retain at least one electrode')
        # Keep the original full-channel tensor and computation route, including
        # in tensor variants: no reshape/selection or completion on this path.
        if bool(mask.all()):
            if not bool(torch.isfinite(raw).all()):
                raise ValueError('Observed EEG contains nonfinite samples')
            return raw, None
        windows = raw.reshape(raw.shape[0], self.channels, self.windows, self.window_samples)
        observed = torch.where(mask[..., None], windows, torch.zeros_like(windows))
        if not bool(torch.isfinite(observed).all()):
            raise ValueError('Observed EEG contains nonfinite samples')
        # The full-channel training/validation path is an exact pass-through.
        # Baselines replace missing normalized samples with zero. Tensor models
        # fill only missing entries; hidden reference values never enter a solve.
        if self.completion is not None:
            observed = self.completion.complete(observed, mask)
        return observed.reshape_as(raw), mask
