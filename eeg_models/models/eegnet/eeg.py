"""The two concrete EEGNet models; MHA is part of their backbone."""
from .backbone import EEGNetBackbone


class EEGNet(EEGNetBackbone):
    model_key = 'eegnet'


class EEGNetTensor(EEGNetBackbone):
    model_key = 'eegnet_tensor'

    def __init__(self, options, metadata, *, window_samples=250, tensor_options=None):
        super().__init__(options, metadata, window_samples=window_samples)
        from .._shared.tensor import Tucker2
        self.signal_input.completion = Tucker2(
            channels=metadata['channels'], features=window_samples, **(tensor_options or {}))
