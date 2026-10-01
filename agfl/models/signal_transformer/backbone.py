"""Shared EEG residual backbone for MHA and Performer."""

import torch
from torch import nn
from .._shared.layers import ElectrodeReadout


def validate_common(options):
    for name in ("dim", "depth"):
        if isinstance(options[name], bool) or not isinstance(options[name], int) or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= options["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if options["mlp_ratio"] <= 0:
        raise ValueError("mlp_ratio must be positive")


class MixerBlock(nn.Module):
    def __init__(self, options, mixer):
        super().__init__()
        dim = options["dim"]
        hidden = max(1, int(dim * options["mlp_ratio"]))
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.dropout = nn.Dropout(options["dropout"])
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(options["dropout"]),
            nn.Linear(hidden, dim), nn.Dropout(options["dropout"]),
        )

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        return x + self.ff(self.norm2(x))


class SignalBackbone(nn.Module):
    def __init__(self, options, metadata, mixer_factory, tokenizer, num_tokens):
        super().__init__()
        validate_common(options)
        self.tokenizer = tokenizer
        self.num_tokens = num_tokens
        self.expected_shape = (int(metadata["channels"]), int(metadata["samples"]))
        self.position = nn.Parameter(torch.empty(1, num_tokens, options["dim"]))
        nn.init.normal_(self.position, std=0.02)
        blocks = []
        for index in range(options['depth']):
            mixer = mixer_factory(options['dim'], num_tokens, index, options['depth'], self.token_axis)
            blocks.append(MixerBlock(options, mixer))
        self.blocks = nn.ModuleList(blocks)
        for index, block in enumerate(self.blocks):
            if hasattr(block.mixer, 'layer_idx'):
                block.mixer.layer_idx = index
                block.mixer.depth = len(self.blocks)
        self.norm = nn.LayerNorm(options["dim"])
        self.classifier = nn.Linear(options["dim"], int(metadata["num_classes"]))
        self.spatial_readout = ElectrodeReadout(num_tokens, options['dim'], options['spatial_readout'])

    def forward_tokens(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != self.expected_shape:
            raise ValueError(f"Expected [B,C,T] with C,T={self.expected_shape}, got {tuple(x.shape)}")
        x = self.tokenizer(x) + self.position
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x):
        tokens = self.forward_tokens(x)
        pooled = self.spatial_readout(tokens)
        return self.classifier(pooled)
