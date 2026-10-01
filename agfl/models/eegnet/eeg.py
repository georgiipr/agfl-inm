"""EEGNet electrode attention with compact or spatial-fusion readout."""
import torch

from .._shared.layers import check_input
from .backbone import EEGNetBackbone


class EEGModel(EEGNetBackbone):
    model_variant = 'eeg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] != 'electrode':
            raise ValueError('EEGNet attention must operate across electrodes')
        super().__init__(options, metadata, attention)

    def forward(self, x):
        check_input(x, self.metadata)
        batch, channels, samples = x.shape
        if self.spatial_fusion:
            # Temporal kernels share parameters; BatchNorm sees one input batch.
            temporal = self.block1(x.unsqueeze(1))
            spatial_features = self.spatial_block3(self.spatial_block2(temporal)).flatten(1)
            electrode_features = temporal.permute(0, 2, 1, 3).reshape(
                batch * channels, temporal.shape[1], 1, samples)
            encoded = self.block3(self.block2(electrode_features))
        else:
            encoded = self.convolve(x.reshape(batch * channels, 1, 1, samples))
        local = self.electrode_projection(encoded.reshape(batch, channels, -1))
        mixed = self.attention_dropout(self.attn_blocks[0](self.electrode_position(local)))
        features = self.spatial_readout(local + mixed if self.attention_residual else mixed)
        if self.spatial_fusion:
            features = torch.cat((spatial_features, features), dim=1)
        return self.fc(features)
