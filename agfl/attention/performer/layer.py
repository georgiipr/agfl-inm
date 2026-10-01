"""FAVOR+ positive orthogonal random-feature softmax approximation.

Choromanski et al., arXiv:2009.14794. Random projections are fixed registered
buffers, seeded on model creation and preserved in checkpoints.
"""
import math
import torch
from ..projections import ProjectedMixer


def orthogonal_features(rows, dim):
    blocks = []
    for _ in range(math.ceil(rows / dim)):
        q, r = torch.linalg.qr(torch.randn(dim, dim))
        q = q * torch.diag(r).sign().unsqueeze(0)
        blocks.append(q.T)
    matrix = torch.cat(blocks)[:rows]
    radii = torch.linalg.vector_norm(torch.randn(rows, dim), dim=-1)
    return matrix * radii[:, None]


class Performer(ProjectedMixer):
    def __init__(self, options):
        super().__init__(options['dim'], options['heads'])
        features = options['random_features']
        if type(features) is not int or features < 1:
            raise ValueError('random_features must be positive')
        self.register_buffer('features', torch.stack([orthogonal_features(features, self.head_dim) for _ in range(self.heads)]))

    def feature_map(self, x, query):
        x = x * self.head_dim ** -.25
        log_features = torch.einsum('bhnd,hmd->bhnm', x, self.features.to(dtype=x.dtype)) - x.square().sum(-1, keepdim=True) / 2
        # Query factors cancel per row; key factors must be common to ALL keys.
        maximum = log_features.amax(dim=-1, keepdim=True) if query else log_features.amax(dim=(-2, -1), keepdim=True)
        return (log_features - maximum.detach()).exp() / self.features.shape[1] ** .5

    def forward(self, x):
        q, k, v = self.project(x)
        original_dtype = v.dtype
        compute_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
        # Exponentials and denominator floors must not underflow in float16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
            q, k = self.feature_map(q, True), self.feature_map(k, False)
            context = k.transpose(-1, -2) @ v
            numerator = q @ context
            denominator = (q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-12)
            result = numerator / denominator
        return self.combine(result.to(original_dtype))
