"""Frozen versus classification-trained spectral Tucker with matched inference.

The historical encoders remain unchanged. Both new arms use the same direct
ridge solve, preprocessing, initial factors, and temporal classification head.
Only the supervised arm exposes its factor matrices to the classifier optimizer.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from inm.tensor_attention import Tucker2
from inm.tensor_temporal.models import TensorTemporal


ARM_IDS = ("spectral_tucker_frozen", "spectral_tucker_supervised")


def observed_ridge_core(values, mask, u, v, ridge=1e-3):
    """Return G[B,Rc,P,Rf] from observed values[B,C,P,F], differentiably.

    Row-major vec(G) has index (r,s). Its normal matrix entry is
    A[r,r'] B[s,s'] + n_observed*F*ridge*I, where A=U_obs.T U_obs
    and B=V.T V. Positive ridge keeps the system nonsingular even for
    repeated factor columns or fewer observed electrodes than channel rank.
    Conversion to float64 and back preserves gradients without eigenvectors.
    """
    if (values.ndim != 4 or values.shape[0] == 0 or values.shape[2] == 0 or
            u.ndim != 2 or v.ndim != 2 or u.shape[1] == 0 or v.shape[1] == 0 or
            (values.shape[1], values.shape[3]) != (u.shape[0], v.shape[0])):
        raise ValueError("Expected values[B,C,P,F] and factors U[C,Rc], V[F,Rf]")
    if (mask.dtype != torch.bool or mask.shape != values.shape[:3] or
            mask.device != values.device or u.device != values.device or v.device != values.device):
        raise ValueError("Boolean availability[B,C,P] and factors must share the input device")
    if (not values.is_floating_point() or values.dtype != u.dtype or values.dtype != v.dtype):
        raise ValueError("Values and factors must share a floating dtype")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if not mask.any(dim=1).all():
        raise ValueError("Every trial/window needs an observed channel")
    safe = torch.where(mask[..., None], values, torch.zeros_like(values))
    if not torch.isfinite(safe).all() or not torch.isfinite(u).all() or not torch.isfinite(v).all():
        raise ValueError("Observed values and factors must be finite")
    batch, channels, windows, features = values.shape
    rows = safe.permute(0, 2, 1, 3).reshape(-1, channels, features).double()
    observed = mask.permute(0, 2, 1).reshape(-1, channels).double()
    u64, v64 = u.double(), v.double()
    a = torch.einsum("cr,nc,ct->nrt", u64, observed, u64)
    b = v64.T @ v64
    rank_channels, rank_features = u.shape[1], v.shape[1]
    dimension = rank_channels * rank_features
    system = torch.einsum("nrt,sq->nrstq", a, b).reshape(-1, dimension, dimension)
    penalty = observed.sum(dim=1) * features * ridge
    system = system + penalty[:, None, None] * torch.eye(dimension, dtype=torch.float64, device=values.device)
    rhs = torch.einsum("cr,ncf,fs->nrs", u64, rows, v64).reshape(-1, dimension, 1)
    core = torch.linalg.solve(system, rhs).reshape(batch, windows, rank_channels, rank_features)
    if not torch.isfinite(core).all():
        raise FloatingPointError("Observed-entry ridge solve produced nonfinite cores")
    return core.permute(0, 2, 1, 3).contiguous().to(values.dtype)


class SpectralTuckerTemporal(TensorTemporal):
    """A matched temporal head with frozen or supervised spectral Tucker factors."""

    def __init__(self, arm_id, seed=0):
        if arm_id not in ARM_IDS:
            raise ValueError(f"Unknown supervised-Tucker arm {arm_id!r}")
        # The parent's constructor dispatches clip_weights before U/V exist.
        super().__init__("fixed_spectral_tucker", seed)
        self.arm_id = arm_id
        u, v = self.tucker.U.clone(), self.tucker.V.clone()
        del self.tucker
        if arm_id == "spectral_tucker_supervised":
            self.U, self.V = nn.Parameter(u), nn.Parameter(v)
        else:
            self.register_buffer("U", u)
            self.register_buffer("V", v)
        self.register_buffer("factor_fit_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("factor_training_seen_counts", torch.zeros(22, dtype=torch.long))

    def constructor_settings(self):
        settings = super().constructor_settings()
        settings["tensor"] = {
            "channel_rank": 4, "spectral_rank": 4, "tucker_ridge": 1e-3,
            "tucker_fit_epochs": 30, "tucker_fit_lr": 0.01, "tucker_fit_batch_size": 64,
            "initialization": "legacy_training_only_reconstruction_fit",
            "inference": "float64_direct_observed_ridge_solve_row_major",
            "ridge_scaling": "observed_channels_times_12_times_ridge",
            "supervised_factors": self.arm_id == "spectral_tucker_supervised",
            "factor_constraint": "supervised_unit_columns_after_optimizer_step",
        }
        return settings

    def calibrate(self, train_raw, *, tucker_epochs=30):
        """Training-only statistics and shared reconstruction initialization.

        Labels and validation arrays are absent from this API. The runner owns
        split isolation. A reduced epoch budget is for synthetic checks only.
        Neither head parameters nor the global RNG state are changed here.
        """
        if type(tucker_epochs) is not int or not 1 <= tucker_epochs <= 30:
            raise ValueError("tucker_epochs must be an integer in 1..30")
        if self.calibrated.item():
            raise RuntimeError("Calibration is immutable; construct a fresh model")
        safe, mask = self._select(train_raw)
        with torch.no_grad():
            features = self._spectral(safe)
            self.spectral_mean.copy_(features.mean(dim=(0, 2), keepdim=True))
            self.spectral_std.copy_(features.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(1e-6))
            standardized = (features - self.spectral_mean) / self.spectral_std
        fitter = Tucker2(22, 12, rank_channels=4, rank_features=4, ridge=1e-3,
                         fit_epochs=30).to(device=safe.device, dtype=safe.dtype)
        devices = [] if safe.device.type == "cpu" else [safe.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.seed + 300007)
            if safe.device.type == "cuda":
                torch.cuda.manual_seed(self.seed + 300007)
            history = fitter.fit(standardized, mask.repeat_interleave(4, dim=2), epochs=tucker_epochs)
        with torch.no_grad():
            self.U.copy_(fitter.U)
            self.V.copy_(fitter.V)
            self.factor_fit_steps.copy_(fitter.fit_steps)
            self.factor_training_seen_counts.copy_(fitter.training_seen_counts)
            self.calibration_trials.fill_(len(safe))
            self.calibrated.fill_(True)
        return {"arm_id": self.arm_id, "training_trials": len(safe),
                "spectral_standardization": True, "factor_history": history,
                "factor_epochs": tucker_epochs,
                "supervised_factors": self.arm_id == "spectral_tucker_supervised"}

    def _encoded(self, selected, mask):
        if not self.calibrated.item():
            raise RuntimeError("Calibrate spectral preprocessing on training trials first")
        expanded = mask.repeat_interleave(4, dim=2)
        values = (self._spectral(selected) - self.spectral_mean) / self.spectral_std
        core = observed_ridge_core(values, expanded, self.U, self.V)
        return core.permute(0, 2, 1, 3).reshape(len(selected), 16, 16)

    @torch.no_grad()
    def clip_weights(self):
        if getattr(self, "arm_id", None) == "spectral_tucker_supervised" and self.calibrated.item():
            for factor in (self.U, self.V):
                Tucker2._unit_columns(factor)


def build_model(arm_id, seed=0):
    return SpectralTuckerTemporal(arm_id, seed)


def restore_model(arm_id, constructor, state_dict):
    if not isinstance(constructor, dict) or "seed" not in constructor:
        raise ValueError("Constructor metadata must include seed")
    model = build_model(arm_id, constructor["seed"])
    if model.constructor_settings() != constructor:
        raise ValueError("Constructor metadata does not match fixed settings")
    model.load_state_dict(state_dict, strict=True)
    return model
