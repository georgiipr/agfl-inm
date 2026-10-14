"""EEGNet and its frozen/learned covariance variants, all with fixed MHA."""
from .eeg import EEGNet, EEGNetFrozenCovariance, EEGNetLearnedCovariance

__all__ = ['EEGNet', 'EEGNetFrozenCovariance', 'EEGNetLearnedCovariance']
