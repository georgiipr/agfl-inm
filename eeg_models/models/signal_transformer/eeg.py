"""Whole-trial temporal encoding per electrode, followed by spatial MHA."""
from torch import nn
from .._shared.pooling import DeterministicAdaptiveAvgPool1d
from .backbone import SignalBackbone, validate_common


class ElectrodeTokenizer(nn.Module):
    def __init__(self, dim, bins, kernel):
        super().__init__()
        if type(bins) is not int or bins < 1 or type(kernel) is not int or kernel < 1 or kernel % 2 == 0:
            raise ValueError('Temporal bins must be positive; kernel size must be positive and odd')
        features = max(4, dim // 4)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, features, kernel, padding=kernel // 2), nn.GELU(),
            DeterministicAdaptiveAvgPool1d(bins), nn.Flatten(), nn.Linear(features * bins, dim))

    def forward(self, x):
        batch, channels, samples = x.shape
        return self.temporal(x.reshape(batch * channels, 1, samples)).reshape(batch, channels, -1)


class SignalTransformer(SignalBackbone):
    model_key = 'signal_transformer'

    def __init__(self, options, metadata, *, window_samples=250):
        validate_common(options)
        tokenizer = ElectrodeTokenizer(options['dim'], options['eeg_temporal_bins'], options['eeg_kernel_size'])
        super().__init__(options, metadata, tokenizer, window_samples=window_samples)


class SignalTransformerFrozenCovariance(SignalTransformer):
    model_key = 'signal_transformer_covariance_frozen'

    def __init__(self, options, metadata, *, window_samples=250, covariance_options=None):
        super().__init__(options, metadata, window_samples=window_samples)
        from .._shared.covariance import CovarianceCompletion
        self.signal_input.completion = CovarianceCompletion(
            channels=metadata['channels'], learned=False, **(covariance_options or {}))


class SignalTransformerLearnedCovariance(SignalTransformer):
    model_key = 'signal_transformer_covariance_learned'

    def __init__(self, options, metadata, *, window_samples=250, covariance_options=None):
        super().__init__(options, metadata, window_samples=window_samples)
        from .._shared.covariance import CovarianceCompletion
        self.signal_input.completion = CovarianceCompletion(
            channels=metadata['channels'], learned=True, **(covariance_options or {}))
