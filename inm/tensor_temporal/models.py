"""EEGNet-free tensor temporal encoders and explicitly defined controls.

Each original one-second window supplies four 100-sample patches at stride 50.
Patches never cross availability boundaries. The first four arms use the same
position-aware 16-token temporal head. The dense spectral arm is a supervised
unstructured compression control, not an isolated test of frozen factorization.
"""
from __future__ import annotations

import torch
from torch import nn

from inm.tensor_attention import Tucker2
from inm.encoder_candidates.transformer import SpatialTransformer, sinusoidal_position_encoding


ARM_IDS = ("fixed_spectral_tucker", "spectral_dense", "learned_tensor_patch",
           "dense_temporal_patch", "spatial_transformer")


class TensorTemporal(nn.Module):
    """Frozen spectral Tucker-2 or supervised multilinear temporal encoder."""

    def __init__(self, arm_id: str, seed: int = 0):
        super().__init__()
        if arm_id not in ARM_IDS[:4]:
            raise ValueError("Unknown tensor-temporal arm")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.arm_id, self.seed = arm_id, seed
        self.register_buffer("hann", torch.hann_window(100, periodic=True))
        self.register_buffer("spectral_mean", torch.zeros(1, 22, 1, 12))
        self.register_buffer("spectral_std", torch.ones(1, 22, 1, 12))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("calibration_trials", torch.tensor(0, dtype=torch.long))
        self.register_buffer("position_encoding", sinusoidal_position_encoding(16, 32))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if arm_id == "fixed_spectral_tucker":
                self.tucker = Tucker2(22, 12, rank_channels=4, rank_features=4,
                                      ridge=1e-3, fit_epochs=30)
            elif arm_id == "spectral_dense":
                self.dense_spectral = nn.Linear(22 * 12, 16)
            else:
                self.U = nn.Parameter(torch.randn(22, 4))
                a = torch.randn(10, 2)
                b = torch.randn(10, 4)
                a = a / torch.linalg.vector_norm(a, dim=0, keepdim=True)
                b = b / torch.linalg.vector_norm(b, dim=0, keepdim=True)
                if arm_id == "learned_tensor_patch":
                    self.A = nn.Parameter(a)
                    self.B = nn.Parameter(b)
                else:
                    # Same initial operator as the factored arm, then free to
                    # leave the Kronecker-structured family during training.
                    self.V = nn.Parameter(torch.einsum("ik,jl->ijkl", a, b).reshape(100, 8))
        # Head initialization does not depend on front-end parameter count.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 200003)
            feature_dim = 16 if arm_id in ARM_IDS[:2] else 32
            self.projection = nn.Linear(feature_dim, 32)
            self.mask_projection = nn.Linear(22, 32, bias=False)
            layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64,
                    dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(32)
            self.classifier = nn.Linear(32 + 88, 4)
        self.clip_weights()

    def constructor_settings(self):
        return {"arm_id": self.arm_id, "seed": self.seed, "version": 1,
                "input": [22, 4, 250], "patch": {"samples": 100, "stride": 50,
                "per_window": 4, "cross_window": False},
                "spectral": {"window": "periodic_hann", "fft_norm": "backward",
                "bins": list(range(1, 13)), "frequencies_hz": [2.5 * i for i in range(1, 13)],
                "power_normalization": "sum_hann_squared", "epsilon": 1e-6,
                "standardization": "training_channel_feature_population_std_floor_1e-6"},
                "tensor": {"channel_rank": 4, "spectral_rank": 4,
                "raw_temporal_shape": [10, 10], "raw_temporal_ranks": [2, 4],
                "tucker_ridge": 1e-3, "tucker_fit_epochs": 30,
                "tucker_fit_lr": 0.01, "tucker_fit_batch_size": 64,
                "raw_nonlinearity": "log_squared_projection_plus_1e-6",
                "dense_temporal_initialization": "matched_kronecker_A_B",
                "factor_constraint": "unit_columns_after_optimizer_step"},
                "head": {"tokens": 16, "d_model": 32, "heads": 4, "layers": 1,
                "feedforward": 64, "dropout": 0.1, "activation": "gelu",
                "norm_first": True, "position": "fixed_sinusoidal",
                "availability": "per_patch_linear22_plus_final88_flags", "pooling": "mean"}}

    def _select(self, raw, mask=None):
        if (not isinstance(raw, torch.Tensor) or raw.ndim != 4 or
                tuple(raw.shape[1:]) != (22, 4, 250) or raw.shape[0] == 0):
            raise ValueError("Raw EEG must be nonempty [B,22,4,250]")
        parameter = self.projection.weight
        if not raw.is_floating_point() or raw.dtype != parameter.dtype or raw.device != parameter.device:
            raise ValueError("Raw EEG must share the model floating dtype and device")
        if mask is None:
            mask = torch.ones(raw.shape[:3], dtype=torch.bool, device=raw.device)
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
                tuple(mask.shape) != tuple(raw.shape[:3]) or mask.device != raw.device):
            raise ValueError("Availability must be Boolean [B,22,4] on the input device")
        if not mask.any(dim=1).all():
            raise ValueError("Every trial/window needs an observed channel")
        safe = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        if not torch.isfinite(safe).all():
            raise ValueError("Observed EEG must be finite")
        return safe, mask

    def _patches(self, selected):
        return selected.unfold(-1, 100, 50).reshape(selected.shape[0], 22, 16, 100)

    def _spectral(self, selected):
        patches = self._patches(selected)
        coefficients = torch.fft.rfft(patches * self.hann, dim=-1)[..., 1:13]
        power = coefficients.abs().square() / self.hann.square().sum()
        return (power + 1e-6).log()

    def calibrate(self, train_raw, *, tucker_epochs=30):
        """Fit preprocessing/frozen factors using fully observed training trials.

        Labels and validation data are intentionally absent from this API.
        A reduced factor budget is available for synthetic software tests only;
        callers must record the returned actual budget in their artifacts.
        """
        if type(tucker_epochs) is not int or not 1 <= tucker_epochs <= 30:
            raise ValueError("tucker_epochs must be an integer in 1..30")
        if self.calibrated.item():
            raise RuntimeError("Calibration is immutable; construct a fresh model")
        safe, mask = self._select(train_raw)
        report = {"arm_id": self.arm_id, "training_trials": len(safe),
                  "spectral_standardization": self.arm_id in ARM_IDS[:2], "factor_history": []}
        if self.arm_id in ARM_IDS[:2]:
            with torch.no_grad():
                features = self._spectral(safe)
                self.spectral_mean.copy_(features.mean(dim=(0, 2), keepdim=True))
                self.spectral_std.copy_(features.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(1e-6))
                standardized = (features - self.spectral_mean) / self.spectral_std
            if self.arm_id == "fixed_spectral_tucker":
                devices = [] if safe.device.type == "cpu" else [safe.device.index]
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(self.seed + 300007)
                    if safe.device.type == "cuda":
                        torch.cuda.manual_seed(self.seed + 300007)
                    report["factor_history"] = self.tucker.fit(
                        standardized, mask.repeat_interleave(4, dim=2), epochs=tucker_epochs)
                report["factor_epochs"] = tucker_epochs
        with torch.no_grad():
            self.calibration_trials.fill_(len(safe))
            self.calibrated.fill_(True)
        return report

    def _encoded(self, selected, mask):
        expanded = mask.repeat_interleave(4, dim=2)
        if self.arm_id in ARM_IDS[:2]:
            if not self.calibrated.item():
                raise RuntimeError("Calibrate spectral preprocessing on training trials first")
            values = (self._spectral(selected) - self.spectral_mean) / self.spectral_std
            values = torch.where(expanded[..., None], values, torch.zeros_like(values))
            if self.arm_id == "fixed_spectral_tucker":
                core = self.tucker.encode(values, expanded)
                return core.permute(0, 2, 1, 3).reshape(len(selected), 16, 16)
            return self.dense_spectral(values.permute(0, 2, 1, 3).reshape(len(selected), 16, 264))
        patches = self._patches(selected)
        if self.arm_id == "learned_tensor_patch":
            core = torch.einsum("bctij,cr,ik,jl->btrkl",
                    patches.reshape(len(selected), 22, 16, 10, 10), self.U, self.A, self.B)
        else:
            core = torch.einsum("bcts,cr,sf->btrf", patches, self.U, self.V)
        return (core.reshape(len(selected), 16, 32).square() + 1e-6).log()

    def forward_features(self, raw, mask=None):
        selected, mask = self._select(raw, mask)
        encoded = self._encoded(selected, mask)
        availability = mask.repeat_interleave(4, dim=2).transpose(1, 2).to(raw.dtype)
        tokens = self.projection(encoded) + self.position_encoding + self.mask_projection(availability)
        return self.final_norm(self.transformer(tokens))

    def forward(self, raw, mask=None):
        selected, availability = self._select(raw, mask)
        features = self.forward_features(selected, availability)
        flags = availability.reshape(len(raw), 88).to(raw.dtype)
        return self.classifier(torch.cat((features.mean(dim=1), flags), dim=1))

    @torch.no_grad()
    def clip_weights(self):
        for name in ("U", "A", "B", "V"):
            factor = getattr(self, name, None)
            if factor is not None:
                factor.div_(torch.linalg.vector_norm(factor, dim=0, keepdim=True).clamp_min(1e-12))


class ReferenceTransformer(SpatialTransformer):
    def calibrate(self, train_raw, *, tucker_epochs=30):
        selected, _ = self._select(train_raw, None)
        return {"arm_id": "spatial_transformer", "training_trials": len(selected),
                "spectral_standardization": False, "factor_history": []}


def build_model(arm_id, seed=0):
    if arm_id not in ARM_IDS:
        raise ValueError(f"Unknown tensor-temporal arm {arm_id!r}")
    return ReferenceTransformer(seed=seed) if arm_id == "spatial_transformer" else TensorTemporal(arm_id, seed)


def restore_model(arm_id, constructor, state_dict):
    if not isinstance(constructor, dict) or "seed" not in constructor:
        raise ValueError("Constructor metadata must include seed")
    model = build_model(arm_id, constructor["seed"])
    if model.constructor_settings() != constructor:
        raise ValueError("Constructor metadata does not match fixed settings")
    model.load_state_dict(state_dict, strict=True)
    return model
