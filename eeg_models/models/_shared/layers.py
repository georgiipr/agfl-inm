"""Fixed MHA, sensor identity and learned spatial readout for both backbones."""
import torch
from torch import nn


def positive_options(options, *names):
    for name in names:
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f'model_options.{name} must be a positive integer')


class MultiHeadAttention(nn.Module):
    """Four-head softmax attention with the original linear-projection initialization.

    Each layer has an independent CPU RNG stream. Its construction leaves the
    surrounding backbone's RNG unchanged; no attention factory is involved.
    """
    def __init__(self, dim, *, layer_index=0):
        super().__init__()
        if dim % 4:
            raise ValueError('MHA dimension must be divisible by four')
        self.dim, self.heads, self.head_dim = dim, 4, dim // 4
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator().manual_seed(
                (torch.initial_seed() + 104729 * (layer_index + 1)) % (2**63))
            torch.set_rng_state(generator.get_state())
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)

    def forward(self, x, padding=None):
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(
            2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) / self.head_dim**.5
        # Full input follows the original MHA operations without a mask branch.
        if padding is not None:
            scores = scores.masked_fill(padding[:, None, None, :], float('-inf'))
        weights = scores.softmax(-1)
        self.last_attn = weights.detach()
        message = weights @ v
        return self.proj(message.transpose(1, 2).reshape(batch, tokens, self.dim))


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
