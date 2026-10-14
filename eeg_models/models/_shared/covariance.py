"""Train-only channel covariance and observed-entry conditional completion.

Frozen and learned variants share initialization and float64 inference. Only
the learned variant exposes the Cholesky coordinates to an optimizer.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def training_second_moment(train_x, *, partition='train'):
    if partition != 'train':
        raise ValueError('Covariance calibration accepts training trials only')
    if (train_x.ndim != 3 or train_x.shape[0] < 1 or train_x.shape[1] != 22
            or not train_x.is_floating_point() or not bool(torch.isfinite(train_x).all())):
        raise ValueError('Expected finite normalized training EEG [N,22,T]')
    rows = train_x.detach().cpu().permute(0, 2, 1).reshape(-1, 22).double()
    moment = rows.T @ rows / len(rows)
    return (moment + moment.T) * .5


class CovarianceCompletion(nn.Module):
    """Fill missing channels with Sigma_HO (Sigma_OO + alpha I)^-1 X_O."""

    def __init__(self, channels=22, *, learned=False, ridge_scale=.001,
                 initialization_jitter=1e-6, diagonal_floor=1e-6):
        super().__init__()
        if channels != 22 or type(learned) is not bool:
            raise ValueError('Completion requires 22 channels and a Boolean learned flag')
        for name, value in (('ridge_scale', ridge_scale),
                            ('initialization_jitter', initialization_jitter),
                            ('diagonal_floor', diagonal_floor)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.channels, self.learned = channels, learned
        self.ridge_scale = float(ridge_scale)
        self.initialization_jitter = float(initialization_jitter)
        self.diagonal_floor = float(diagonal_floor)
        coordinates = torch.zeros(channels * (channels + 1) // 2, dtype=torch.float64)
        if learned:
            self.free = nn.Parameter(coordinates)
        else:
            self.register_buffer('free', coordinates)
        self.register_buffer('initial_free', coordinates.clone())
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def initialize(self, second_moment, *, partition='train'):
        if partition != 'train':
            raise ValueError('Covariance initialization is training-only')
        moment = second_moment.detach().to(device=self.free.device, dtype=torch.float64)
        if (moment.shape != (self.channels, self.channels)
                or not bool(torch.isfinite(moment).all())
                or not torch.allclose(moment, moment.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('Expected a finite symmetric training second moment')
        lower = torch.linalg.cholesky(moment + self.initialization_jitter * torch.eye(
            self.channels, device=moment.device, dtype=moment.dtype))
        indices = torch.tril_indices(self.channels, self.channels, device=moment.device)
        values = lower[indices[0], indices[1]].clone()
        diagonal = indices[0] == indices[1]
        positive = values[diagonal] - self.diagonal_floor
        if bool((positive <= 0).any()):
            raise ValueError('Cholesky diagonal must exceed diagonal_floor')
        values[diagonal] = positive + torch.log(-torch.expm1(-positive))
        self.free.copy_(values)
        self.initial_free.copy_(values)
        self.fitted.fill_(True)

    def covariance(self):
        if not bool(self.fitted) or self.free.dtype != torch.float64:
            raise RuntimeError('Load or fit training-only float64 covariance before completion')
        if not bool(torch.isfinite(self.free).all()):
            raise FloatingPointError('Nonfinite covariance parameters')
        indices = torch.tril_indices(self.channels, self.channels, device=self.free.device)
        values = torch.where(indices[0] == indices[1],
                             F.softplus(self.free) + self.diagonal_floor, self.free)
        lower = self.free.new_zeros(self.channels, self.channels).index_put(
            (indices[0], indices[1]), values)
        sigma = lower @ lower.T
        if not bool(torch.isfinite(sigma).all()):
            raise FloatingPointError('Nonfinite covariance matrix')
        return sigma

    def complete(self, x, mask):
        if (x.ndim != 4 or x.shape[0] < 1 or x.shape[1] != self.channels
                or x.shape[2] < 1 or x.shape[3] < 1 or not x.is_floating_point()
                or x.device != self.free.device):
            raise ValueError('Expected EEG windows [B,22,P,F] on the covariance device')
        if (mask.dtype != torch.bool or mask.shape != x.shape[:3] or mask.device != x.device
                or not bool(mask.any(dim=1).all())):
            raise ValueError('Expected Boolean [B,22,P] with observed channels in every window')
        safe = torch.where(mask[..., None], x, torch.zeros_like(x))
        if not bool(torch.isfinite(safe).all()):
            raise ValueError('Observed EEG contains nonfinite values')
        if bool(mask.all()):
            return x
        sigma = self.covariance()
        ridge = self.ridge_scale * sigma.diag().mean()
        if not bool(torch.isfinite(ridge)) or float(ridge.detach()) <= 0:
            raise FloatingPointError('Invalid covariance ridge')
        batch, _, windows, samples = x.shape
        rows = safe.permute(0, 2, 1, 3).reshape(-1, self.channels, samples).double()
        flags = mask.permute(0, 2, 1).reshape(-1, self.channels)
        estimates = torch.zeros_like(rows)
        patterns, inverse = torch.unique(flags, dim=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            known, missing = torch.nonzero(pattern).flatten(), torch.nonzero(~pattern).flatten()
            if not len(missing):
                continue
            selected = torch.nonzero(inverse == index).flatten()
            system = sigma[known][:, known] + ridge * torch.eye(
                len(known), device=x.device, dtype=torch.float64)
            beta = torch.linalg.solve(system, sigma[missing][:, known].T)
            prediction = torch.einsum('mo,not->nmt', beta.T, rows[selected][:, known])
            estimates[selected[:, None], missing[None, :]] = prediction
        estimate = estimates.reshape(batch, windows, self.channels, samples).permute(0, 2, 1, 3).to(x.dtype)
        if not bool(torch.isfinite(estimate).all()):
            raise FloatingPointError('Nonfinite covariance completion')
        return torch.where(mask[..., None], x, estimate)

    forward = complete

    def anchor_penalty(self):
        return (self.free - self.initial_free).square().mean()
