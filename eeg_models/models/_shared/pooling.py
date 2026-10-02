"""Adaptive means with deterministic CUDA backward and native bin boundaries."""
import torch
from torch import nn


class DeterministicAdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        if type(output_size) is not int or output_size < 1:
            raise ValueError('output_size must be a positive integer')
        self.output_size = output_size

    def forward(self, x):
        if x.ndim not in (2, 3) or x.shape[-1] < 1:
            raise ValueError('Expected [C,T] or [B,C,T] with nonempty time axis')
        length, bins = x.shape[-1], self.output_size
        return torch.stack([x[..., i * length // bins:((i + 1) * length + bins - 1) // bins].mean(-1)
                            for i in range(bins)], dim=-1)
