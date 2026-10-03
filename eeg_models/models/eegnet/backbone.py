"""EEGNet temporal -> fixed MHA -> spatial -> separable -> classifier route."""
import math
import torch
from torch import nn
from .._shared.input import SignalInput
from .._shared.layers import ElectrodeIdentity, MultiHeadAttention, positive_options


class EEGNetBackbone(nn.Module):
    token_axis = 'electrode'

    def __init__(self, options, metadata, *, window_samples=250):
        super().__init__()
        positive_options(options, 'temp_kernel', 'f1', 'd', 'f2', 'pk1', 'pk2')
        if options['f1'] % 4:
            raise ValueError('EEGNet f1 must be divisible by the four built-in MHA heads')
        if not 0 <= options['dropout_rate'] < 1 or min(options['max_norm1'], options['max_norm2']) <= 0:
            raise ValueError('Invalid EEGNet dropout or max norm')
        momentum, eps = options['batch_norm_momentum'], options['batch_norm_eps']
        if not math.isfinite(momentum) or not 0 < momentum <= 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError('Invalid EEGNet BatchNorm settings')
        self.metadata = metadata
        self.num_tokens = metadata['channels']
        self.signal_input = SignalInput(metadata, window_samples)
        self.max_norm1, self.max_norm2 = options['max_norm1'], options['max_norm2']
        f1, expanded, f2 = options['f1'], options['f1'] * options['d'], options['f2']
        def bn(features):
            return nn.BatchNorm2d(features, momentum=momentum, eps=eps)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, options['temp_kernel']), padding='same', bias=False), bn(f1))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, expanded, (self.num_tokens, 1), groups=f1, bias=False),
            bn(expanded), nn.ELU(), nn.AvgPool2d((1, options['pk1'])),
            nn.Dropout(options['dropout_rate']))
        self.block3 = nn.Sequential(
            nn.Conv2d(expanded, expanded, (1, 16), padding='same', groups=expanded, bias=False),
            nn.Conv2d(expanded, f2, 1, bias=False), bn(f2), nn.ELU(),
            nn.AvgPool2d((1, options['pk2'])), nn.Dropout(options['dropout_rate']))
        steps = metadata['samples'] // options['pk1'] // options['pk2']
        if steps < 1:
            raise ValueError('EEGNet pooling leaves no temporal samples')
        # Preserve the original convolution -> attention -> classifier
        # construction order and independent attention initialization stream.
        self.electrode_position = ElectrodeIdentity(f1, self.num_tokens)
        self.attn_blocks = nn.ModuleList([MultiHeadAttention(f1)])
        self.attention_dropout = nn.Dropout(0.0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(f2 * steps, metadata['num_classes'])
        self.clip_weights()

    @property
    def classifier(self):
        return self.fc

    def forward(self, raw, mask=None):
        raw, mask = self.signal_input(raw, mask)
        batch, channels, samples = raw.shape
        temporal = self.block1(raw.unsqueeze(1))  # B,F1,C,T: whole-trial temporal encoding
        tokens = temporal.permute(0, 3, 2, 1).reshape(batch * samples, channels, -1)
        positioned = self.electrode_position(tokens)
        padding = None
        if mask is not None:
            observed = mask.repeat_interleave(self.signal_input.window_samples, dim=2)
            padding = ~observed.transpose(1, 2).reshape(batch * samples, channels)
        mixed = self.attention_dropout(self.attn_blocks[0](positioned, padding))
        mixed = mixed.reshape(batch, samples, channels, -1).permute(0, 3, 2, 1)
        signal = temporal + mixed
        return self.fc(self.flatten(self.block3(self.block2(signal))))

    @torch.no_grad()
    def clip_weights(self):
        for layer, maximum in ((self.block2[0], self.max_norm1), (self.fc, self.max_norm2)):
            layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=maximum))
