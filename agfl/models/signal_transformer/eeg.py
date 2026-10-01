"""EEG: temporal encoding per electrode, then mixing across electrodes."""

from torch import nn
from .._shared.pooling import DeterministicAdaptiveAvgPool1d

from .backbone import SignalBackbone, validate_common


class ElectrodeTokenizer(nn.Module):
    def __init__(self, dim, bins, kernel):
        super().__init__()
        if not isinstance(bins, int) or bins < 1 or not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0:
            raise ValueError("eeg_temporal_bins must be positive; eeg_kernel_size must be positive and odd")
        features = max(4, dim // 4)
        # Shared temporal kernel; channels are never convolved together here.
        self.temporal = nn.Sequential(
            nn.Conv1d(1, features, kernel, padding=kernel // 2), nn.GELU(),
            DeterministicAdaptiveAvgPool1d(bins), nn.Flatten(), nn.Linear(features * bins, dim),
        )

    def forward(self, x):
        batch, channels, samples = x.shape
        return self.temporal(x.reshape(batch * channels, 1, samples)).reshape(batch, channels, -1)


class EEGSignalTransformer(SignalBackbone):
    token_axis = "electrode"
    model_variant = "eeg"

    def __init__(self, options, metadata, mixer_factory):
        validate_common(options)
        tokenizer = ElectrodeTokenizer(options["dim"], options["eeg_temporal_bins"], options["eeg_kernel_size"])
        super().__init__(options, metadata, mixer_factory, tokenizer, int(metadata["channels"]))


EEGModel = EEGSignalTransformer
