"""Matching learned Q/K/V and output projections for attention baselines."""

from torch import nn


class ProjectedMixer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def project(self, x):
        batch, tokens, _ = x.shape
        return self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

    def combine(self, x):
        batch, _, tokens, _ = x.shape
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, self.dim))
