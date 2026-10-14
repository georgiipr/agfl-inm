"""Signal Transformer and its covariance variants, all with fixed MHA."""
from .eeg import SignalTransformer, SignalTransformerFrozenCovariance, SignalTransformerLearnedCovariance

__all__ = ['SignalTransformer', 'SignalTransformerFrozenCovariance', 'SignalTransformerLearnedCovariance']
