"""Lazy factory and checkpoint reconstruction for candidate arms."""
from __future__ import annotations


LOCAL_ARMS = ("local_control", "local_power")
SPATIAL_ARMS = ("spatial_eegnet", "spatial_filterbank")
TRANSFORMER_ARMS = ("spatial_transformer",)


def build_model(arm_id, seed):
    """Construct an arm; imports for not-yet-implemented arms remain lazy."""
    if arm_id in LOCAL_ARMS:
        from .local import LocalClassifier
        return LocalClassifier(arm_id, seed=seed)
    if arm_id == "spatial_eegnet":
        from .spatial import SpatialEEGNet
        return SpatialEEGNet(seed=seed)
    if arm_id == "spatial_filterbank":
        from .spatial import SpatialFilterBank
        return SpatialFilterBank(seed=seed)
    if arm_id == "spatial_transformer":
        from .transformer import SpatialTransformer
        return SpatialTransformer(seed=seed)
    raise ValueError(f"Unknown encoder-candidate arm {arm_id!r}")


def restore_model(arm_id, constructor, state_dict):
    """Rebuild an implemented arm from primitive constructor metadata."""
    if not isinstance(constructor, dict):
        raise ValueError("Constructor metadata must be a dictionary")
    if arm_id in LOCAL_ARMS:
        if constructor.get("kind") != arm_id:
            raise ValueError("Constructor metadata does not match the requested arm")
        from .local import LocalClassifier
        if set(constructor) != {"kind", "seed", "encoder", "head"}:
            raise ValueError("Constructor metadata has unknown or missing settings")
        model = LocalClassifier(arm_id, seed=constructor["seed"])
    elif arm_id in SPATIAL_ARMS:
        from .spatial import SpatialEEGNet, SpatialFilterBank
        cls = SpatialEEGNet if arm_id == "spatial_eegnet" else SpatialFilterBank
        expected_keys = ({"seed", "encoder"} if arm_id == "spatial_eegnet" else
                         {"seed", "channels", "windows", "window_samples", "bands_hz",
                          "fir_taps", "fir_window", "sampling_rate",
                          "spatial_filters_per_band", "spatial_filter_max_norm",
                          "log_epsilon", "dropout", "num_classes"})
        if set(constructor) != expected_keys:
            raise ValueError("Constructor metadata has unknown or missing settings")
        if arm_id == "spatial_eegnet":
            if constructor["encoder"] != cls._fixed_encoder_settings():
                raise ValueError("Constructor metadata does not match the implemented model settings")
            model = cls(seed=constructor["seed"])
        else:
            settings = dict(constructor)
            settings.pop("seed")
            model = cls(seed=constructor["seed"], **settings)
    elif arm_id in TRANSFORMER_ARMS:
        from .transformer import SpatialTransformer
        expected_keys = {"seed", "encoder", "transformer", "num_classes", "mask_flags"}
        if set(constructor) != expected_keys:
            raise ValueError("Constructor metadata has unknown or missing settings")
        model = SpatialTransformer(seed=constructor["seed"])
    else:
        raise ValueError(f"Unknown encoder-candidate arm {arm_id!r}")
    if model.constructor_settings() != constructor:
        raise ValueError("Constructor metadata does not match the implemented model settings")
    model.load_state_dict(state_dict, strict=True)
    return model
