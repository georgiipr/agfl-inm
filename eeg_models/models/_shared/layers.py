"""Sensor identity and fixed learned spatial readout used by the backbones."""
import torch
from torch import nn


def positive_options(options, *names):
    for name in names:
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f'model_options.{name} must be a positive integer')


class ElectrodeIdentity(nn.Module):
    """Sensor identities, with the same channel order in every trial/checkpoint."""
    def __init__(self, dim, channels):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, channels, dim))

    def forward(self, x):
        if x.shape[1:] != self.embedding.shape[1:]:
            raise ValueError('Electrode identity shape must match channel tokens')
        return x + self.embedding


class ElectrodeReadout(nn.Module):
    """One learned signed spatial filter per feature."""
    def __init__(self, channels, features):
        super().__init__()
        self.projection = nn.Conv1d(features, features, channels, groups=features, bias=False)

    def forward(self, tokens):
        return self.projection(tokens.transpose(1, 2)).squeeze(-1)
