"""Training-fitted Tucker-2 factors and mask-aware feature/core inference.

This file is deliberately independent of the classifier and experiment runner.
``X`` has shape [batch, channels, windows, features]; its boolean
availability mask has shape [batch, channels, windows]. A hypermatrix here is
the same higher-order array as a tensor, not an additional learned mechanism.

Only training-set features may be passed to ``fit``. Factors are stored as
buffers, not Parameters, so a classifier optimizer cannot silently update them.
See ``docs/tensor_math.md`` for the objective, normal equations, and limitations.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Tucker2(nn.Module):
    """Shared channel/feature factors with exact regularized core inference.

    Seed PyTorch before constructing/fitting the object for reproducible factor
    initialization. Move the object to the feature device with ``.to(device)``.
    The temporal axis is retained; its length can change between calls.

    ``fit`` performs alternating core solves and projected factor-gradient
    updates. Once fitted, ``encode`` and ``reconstruct`` never update factors,
    read labels, or use unavailable feature entries. Checkpoints should include
    this module's ``state_dict`` along with training-only preprocessing state.
    """

    def __init__(
        self,
        channels: int,
        features: int,
        rank_channels: int = 4,
        rank_features: int = 4,
        ridge: float = 1e-3,
        fit_epochs: int = 30,
        fit_lr: float = 1e-2,
        fit_batch_size: int = 64,
    ) -> None:
        super().__init__()
        dimensions = {
            "channels": channels,
            "features": features,
            "rank_channels": rank_channels,
            "rank_features": rank_features,
            "fit_epochs": fit_epochs,
            "fit_batch_size": fit_batch_size,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if rank_channels > channels or rank_features > features:
            raise ValueError("Tucker ranks cannot exceed their channel/feature dimensions")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and strictly positive")
        if not math.isfinite(fit_lr) or fit_lr <= 0:
            raise ValueError("fit_lr must be finite and strictly positive")
        self.channels = channels
        self.features = features
        self.rank_channels = rank_channels
        self.rank_features = rank_features
        self.ridge = float(ridge)
        self.fit_epochs = fit_epochs
        self.fit_lr = float(fit_lr)
        self.fit_batch_size = fit_batch_size
        self.register_buffer("U", torch.zeros(channels, rank_channels))
        self.register_buffer("V", torch.zeros(features, rank_features))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("fit_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("training_seen_counts", torch.zeros(channels, dtype=torch.long))

    def _available_values(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or tuple(x.shape[1::2]) != (self.channels, self.features):
            raise ValueError(
                f"Expected X[B,{self.channels},P,{self.features}], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0 or x.shape[2] == 0:
            raise ValueError("At least one recording and one temporal window are required")
        if not torch.is_floating_point(x):
            raise TypeError("Features must be floating-point tensors")
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:3]):
            raise ValueError("Availability mask must be boolean with shape [B,C,P]")
        if x.device != self.U.device or mask.device != x.device:
            raise ValueError("Features, availability mask, and Tucker factors must share a device")
        if (mask.sum(dim=1) == 0).any().item():
            raise ValueError("Every recording/window must have at least one observed channel")
        # IMPORTANT: never multiply raw unavailable values by zero. They may be
        # NaN, Inf, or arbitrary placeholders; none may enter a solve or loss.
        safe = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        if not torch.isfinite(safe).all().item():
            raise ValueError("Observed feature values must be finite")
        return safe, mask

    @staticmethod
    @torch.no_grad()
    def _unit_columns(factor: Tensor) -> None:
        norms = torch.linalg.vector_norm(factor, dim=0, keepdim=True)
        if not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
            raise FloatingPointError("Tucker factor update produced invalid or zero columns")
        factor.div_(norms)

    @torch.no_grad()
    def _solve(self, safe: Tensor, mask: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Solve A G B + (n_observed * F * ridge) G = C in float64.

        The two small symmetric eigensystems avoid materializing a Kronecker
        design matrix. Positive ridge makes the core unique even when fewer
        channels are observed than the channel rank. No autograd graph is built.
        """
        batch, _, windows, _ = safe.shape
        rows = safe.permute(0, 2, 1, 3).reshape(-1, self.channels, self.features)
        observed = mask.permute(0, 2, 1).reshape(-1, self.channels)
        u64, v64 = u.to(torch.float64), v.to(torch.float64)
        rows64 = rows.to(torch.float64)
        observed64 = observed.to(torch.float64)
        a = torch.einsum("cr,nc,cs->nrs", u64, observed64, u64)
        b = v64.T @ v64
        rhs = torch.einsum("cr,ncf,fs->nrs", u64, rows64, v64)
        a = (a + a.transpose(-1, -2)) * 0.5
        b = (b + b.T) * 0.5
        eigen_a, qa = torch.linalg.eigh(a)
        eigen_b, qb = torch.linalg.eigh(b)
        # These Gram matrices are positive semidefinite; remove only numerical
        # roundoff below zero before forming the strictly positive denominator.
        eigen_a = eigen_a.clamp_min(0)
        eigen_b = eigen_b.clamp_min(0)
        scale = observed64.sum(dim=1) * self.features * self.ridge
        denominator = eigen_a.unsqueeze(-1) * eigen_b.view(1, 1, -1)
        denominator = denominator + scale.view(-1, 1, 1)
        rotated_rhs = qa.transpose(-1, -2) @ rhs @ qb
        core = qa @ (rotated_rhs / denominator) @ qb.T
        if not torch.isfinite(core).all().item():
            raise FloatingPointError("Tucker ridge solve produced non-finite core values")
        core = core.reshape(batch, windows, self.rank_channels, self.rank_features)
        return core.permute(0, 2, 1, 3).contiguous().to(dtype=u.dtype)

    @staticmethod
    def _estimate(core: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return torch.einsum("cr,brps,fs->bcpf", u, core, v)

    def _objective(
        self, safe: Tensor, mask: Tensor, core: Tensor, u: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        estimate = self._estimate(core, u, v)
        residual = torch.where(mask.unsqueeze(-1), estimate - safe, torch.zeros_like(safe))
        denominator = mask.sum(dim=1).to(dtype=safe.dtype) * self.features
        reconstruction = (residual.square().sum(dim=(1, 3)) / denominator).mean()
        core_penalty = self.ridge * core.square().sum(dim=(1, 3)).mean()
        return reconstruction + core_penalty, reconstruction, core_penalty

    def fit(
        self,
        train_x: Tensor,
        train_mask: Tensor,
        *,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
    ) -> list[dict[str, float | int]]:
        """Fit factors using TRAINING recordings and their actual input masks.

        Returns post-epoch training-only objective diagnostics. Validation/test
        arrays must never be supplied, including for initialization. The caller
        owns split isolation; this low-level tensor API cannot infer split roles.
        No classifier parameters are optimized here. Calling fit again explicitly
        starts a new factor fit and invalidates the previous fitted state.
        """
        safe, mask = self._available_values(train_x, train_mask)
        safe = safe.detach().to(dtype=self.U.dtype)
        mask = mask.detach()
        epochs = self.fit_epochs if epochs is None else epochs
        lr = self.fit_lr if lr is None else lr
        batch_size = self.fit_batch_size if batch_size is None else batch_size
        for name, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Factor learning rate must be finite and positive")
        counts = mask.sum(dim=(0, 2))
        if (counts == 0).any().item():
            unseen = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"Cannot identify factor rows for channels never seen in training: {unseen}")
        self.fitted.fill_(False)
        self.fit_steps.zero_()
        self.training_seen_counts.copy_(counts)
        # Ephemeral gradient-bearing tensors are local to fitting. The model's
        # persistent U/V buffers stay outside every classifier optimizer.
        u = torch.randn_like(self.U)
        v = torch.randn_like(self.V)
        self._unit_columns(u)
        self._unit_columns(v)
        u.requires_grad_(True)
        v.requires_grad_(True)
        optimizer = torch.optim.Adam([u, v], lr=lr)
        history: list[dict[str, float | int]] = []
        for epoch in range(epochs):
            permutation = torch.randperm(safe.shape[0], device=safe.device)
            for start in range(0, safe.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                batch_x, batch_mask = safe[indices], mask[indices]
                core = self._solve(batch_x, batch_mask, u.detach(), v.detach())
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = self._objective(batch_x, batch_mask, core, u, v)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite training-only Tucker objective")
                loss.backward()
                optimizer.step()
                self._unit_columns(u)
                self._unit_columns(v)
            totals = [0.0, 0.0, 0.0]
            with torch.no_grad():
                # Re-solve at the current factors, so history reports the actual
                # post-update objective rather than stale minibatch cores.
                for start in range(0, safe.shape[0], batch_size):
                    batch_x = safe[start : start + batch_size]
                    batch_mask = mask[start : start + batch_size]
                    core = self._solve(batch_x, batch_mask, u, v)
                    values = self._objective(batch_x, batch_mask, core, u, v)
                    for index, value in enumerate(values):
                        totals[index] += float(value.item()) * len(batch_x)
            history.append({
                "epoch": epoch + 1,
                "objective": totals[0] / len(safe),
                "reconstruction_mse": totals[1] / len(safe),
                "core_penalty": totals[2] / len(safe),
            })
        with torch.no_grad():
            self.U.copy_(u.detach())
            self.V.copy_(v.detach())
            self.fit_steps.fill_(epochs)
            self.fitted.fill_(True)
        return history

    @torch.no_grad()
    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        """Infer G[B,Rc,P,Rf] from observed input values, with factors frozen."""
        if not self.fitted.item():
            raise RuntimeError("Fit training-only Tucker factors or load a fitted state_dict first")
        safe, mask = self._available_values(x, mask)
        return self._solve(safe, mask, self.U, self.V).to(dtype=x.dtype)

    @torch.no_grad()
    def reconstruct(self, x: Tensor, mask: Tensor) -> Tensor:
        """Estimate all feature entries; only available inputs enter inference."""
        core = self.encode(x, mask).to(dtype=self.U.dtype)
        return self._estimate(core, self.U, self.V).to(dtype=x.dtype)

    @torch.no_grad()
    def complete(self, x: Tensor, mask: Tensor) -> Tensor:
        """Fill missing features and preserve every observed feature exactly."""
        estimate = self.reconstruct(x, mask)
        return torch.where(mask.unsqueeze(-1), x, estimate)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.encode(x, mask)
