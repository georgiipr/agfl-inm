"""Defaults shared by comparable mechanisms, independent of the dataset."""

DEFAULTS = {
    "dim": 64,
    "depth": 2,
    "dropout": 0.1,
    "mlp_ratio": 4.0,
    "eeg_temporal_bins": 8,
    "eeg_kernel_size": 15,
    "spatial_readout": "learned",
}
