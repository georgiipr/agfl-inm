"""Standard scaled dot-product multi-head attention over spatial tokens."""
from ..projections import ProjectedMixer


class MultiHeadAttention(ProjectedMixer):
    def __init__(self, dim, heads):
        super().__init__(dim, heads)
        self.options = {'dim': dim, 'heads': heads}

    def forward(self, x):
        q, k, v = self.project(x)
        scores = q @ k.transpose(-1, -2) / self.head_dim**.5
        weights = scores.softmax(-1)
        self.last_attn = weights.detach()
        return self.combine(weights @ v)
