"""Signal Transformer with fixed MHA in every residual block."""
import torch
from torch import nn
from .._shared.input import SignalInput
from .._shared.layers import ElectrodeReadout, MultiHeadAttention, positive_options


def validate_common(options):
    positive_options(options, 'dim', 'depth')
    if options['dim'] % 4 or not 0 <= options['dropout'] < 1 or options['mlp_ratio'] <= 0:
        raise ValueError('Invalid Signal Transformer dimension/dropout/MLP settings')


class TransformerBlock(nn.Module):
    def __init__(self, options, layer_index):
        super().__init__()
        dim = options['dim']
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MultiHeadAttention(dim, layer_index=layer_index)
        self.dropout = nn.Dropout(options['dropout'])
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(1, int(dim * options['mlp_ratio']))
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(options['dropout']),
            nn.Linear(hidden, dim), nn.Dropout(options['dropout']))

    def forward(self, x, padding):
        normalized = self.norm1(x)
        mixed = self.mixer(normalized, padding)
        x = x + self.dropout(mixed)
        return x + self.ff(self.norm2(x))


class SignalBackbone(nn.Module):
    token_axis = 'electrode'

    def __init__(self, options, metadata, tokenizer, *, window_samples=250):
        super().__init__()
        validate_common(options)
        self.signal_input = SignalInput(metadata, window_samples)
        self.tokenizer = tokenizer
        self.num_tokens = metadata['channels']
        self.position = nn.Parameter(torch.empty(1, self.num_tokens, options['dim']))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(options, index) for index in range(options['depth'])])
        self.norm = nn.LayerNorm(options['dim'])
        self.classifier = nn.Linear(options['dim'], metadata['num_classes'])
        self.spatial_readout = ElectrodeReadout(self.num_tokens, options['dim'])

    def forward(self, raw, mask=None):
        raw, mask = self.signal_input(raw, mask)
        tokens = self.tokenizer(raw) + self.position
        # A token summarizes the full trial of one electrode. A channel with
        # any observed window is a key; an entirely absent channel is not.
        # The same rule applies to measured and tensor-completed inputs.
        padding = None if mask is None else ~mask.any(dim=2)
        for block in self.blocks:
            tokens = block(tokens, padding)
        return self.classifier(self.spatial_readout(self.norm(tokens)))
