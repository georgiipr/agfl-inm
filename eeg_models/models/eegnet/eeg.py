"""EEGNet and two covariance variants; MHA is part of the backbone."""
from .backbone import EEGNetBackbone


class EEGNet(EEGNetBackbone):
    model_key = 'eegnet'


class EEGNetFrozenCovariance(EEGNetBackbone):
    model_key = 'eegnet_covariance_frozen'

    def __init__(self, options, metadata, *, window_samples=250, covariance_options=None):
        super().__init__(options, metadata, window_samples=window_samples)
        from .._shared.covariance import CovarianceCompletion
        self.signal_input.completion = CovarianceCompletion(
            channels=metadata['channels'], learned=False, **(covariance_options or {}))


class EEGNetLearnedCovariance(EEGNetBackbone):
    model_key = 'eegnet_covariance_learned'

    def __init__(self, options, metadata, *, window_samples=250, covariance_options=None):
        super().__init__(options, metadata, window_samples=window_samples)
        from .._shared.covariance import CovarianceCompletion
        self.signal_input.completion = CovarianceCompletion(
            channels=metadata['channels'], learned=True, **(covariance_options or {}))
