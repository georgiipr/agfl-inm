"""Input contracts, electrode identity embeddings, and spatial readout."""
import torch
from torch import nn


def check_input(x, metadata):
    expected = (int(metadata['channels']), int(metadata['samples']))
    if x.ndim != 3 or tuple(x.shape[1:]) != expected:
        raise ValueError(f'Expected [B,C,T] with C,T={expected}, got {tuple(x.shape)}')


def positive_options(options, *names):
    for name in names:
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f'model_options.{name} must be a positive integer')


class ElectrodeIdentity(nn.Module):
    """Learn sensor identities without treating channel-list gaps as distance.

    Fixed input channel order is part of dataset/checkpoint provenance. These
    embeddings are not coordinates or anatomical brain-region assignments.
    """
    def __init__(self, dim, channels):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, channels, dim))

    def forward(self, x):
        if x.shape[1:] != self.embedding.shape[1:]:
            raise ValueError('Electrode identity shape must match the channel feature tensor')
        return x + self.embedding


class ElectrodeReadout(nn.Module):
    """Learn a signed spatial filter per feature, retaining electrode identity."""
    def __init__(self, channels, features, mode='learned'):
        super().__init__()
        if mode not in {'learned', 'mean'}:
            raise ValueError('spatial_readout must be learned or mean')
        self.projection = (nn.Conv1d(features, features, channels, groups=features, bias=False)
                           if mode == 'learned' else None)

    def forward(self, tokens):
        if self.projection is None:
            return tokens.mean(dim=1)
        return self.projection(tokens.transpose(1, 2)).squeeze(-1)
