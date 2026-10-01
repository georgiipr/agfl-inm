"""EEGNet-derived window features and interchangeable spatial-attention heads.

This is a new EEGNet adaptation, not the full-channel ``spatial_fusion`` model
in the EEG model library. The feature extractor cannot read an unavailable raw
window. Attention is spatial within each window; it never runs over time.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from agfl.attention import AttentionFactory


def _positive(**values):
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')


def _availability(mask, *, batch, channels, windows, device):
    if mask is None:
        return torch.ones(batch, channels, windows, device=device, dtype=torch.bool)
    if tuple(mask.shape) != (batch, channels, windows) or mask.dtype != torch.bool:
        raise ValueError('Availability must be boolean [batch, channels, windows]')
    if mask.device != device:
        raise ValueError('Availability and input must be on the same device')
    if not mask.any(1).all():
        raise ValueError('Every trial/window must contain at least one observed channel')
    return mask


class EEGWindowEncoder(nn.Module):
    """Share EEGNet channel-local convolution weights over separate windows.

    Input is [B,C,P*W], output [B,C,P,F]. Defaults give C=22, P=4, W=250,
    F=32. A missing raw window is replaced *before* the first convolution and
    its output is zeroed afterward. Frozen BatchNorm uses training statistics.
    """

    def __init__(self, *, channels=22, windows=4, window_samples=250,
                 f1=16, d=2, f2=32, kernel_length=32, pool1=8, pool2=16,
                 dropout=.5):
        super().__init__()
        _positive(channels=channels, windows=windows, window_samples=window_samples,
                  f1=f1, d=d, f2=f2, kernel_length=kernel_length, pool1=pool1, pool2=pool2)
        if not 0 <= dropout < 1:
            raise ValueError('dropout must lie in [0, 1)')
        steps = window_samples // pool1 // pool2
        if steps < 1:
            raise ValueError('The EEGNet pooling factors leave an empty window')
        self.channels, self.windows, self.window_samples = channels, windows, window_samples
        self.features = f2 * steps
        self.frozen = False
        expanded = f1 * d
        # These are the three channel-local blocks of EEGNetBackbone in the
        # retained model library, instantiated without its full-trial readout/attention.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(f1, momentum=.1, eps=1e-5))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, expanded, (1, 1), groups=f1, bias=False),
            nn.BatchNorm2d(expanded, momentum=.1, eps=1e-5), nn.ELU(),
            nn.AvgPool2d((1, pool1)), nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Conv2d(expanded, expanded, (1, 16), padding='same',
                      groups=expanded, bias=False),
            nn.Conv2d(expanded, f2, 1, bias=False),
            nn.BatchNorm2d(f2, momentum=.1, eps=1e-5), nn.ELU(),
            nn.AvgPool2d((1, pool2)), nn.Dropout(dropout))
        self.clip_weights()

    def forward(self, raw, mask=None):
        expected = (self.channels, self.windows * self.window_samples)
        if raw.ndim != 3 or tuple(raw.shape[1:]) != expected:
            raise ValueError(f'Raw EEG must have shape [B,{expected[0]},{expected[1]}]')
        batch = raw.shape[0]
        mask = _availability(mask, batch=batch, channels=self.channels,
                             windows=self.windows, device=raw.device)
        windows = raw.reshape(batch, self.channels, self.windows, self.window_samples)
        # Multiplication would leave NaNs in hidden locations; selection does not.
        observed = torch.where(mask[..., None], windows, torch.zeros_like(windows))
        if not torch.isfinite(observed).all():
            raise ValueError('Observed raw EEG contains non-finite values')
        separated = observed.reshape(-1, 1, 1, self.window_samples)
        with torch.no_grad() if self.frozen else nullcontext():
            encoded = self.block3(self.block2(self.block1(separated)))
        encoded = encoded.reshape(batch, self.channels, self.windows, self.features)
        return torch.where(mask[..., None], encoded, torch.zeros_like(encoded))

    def freeze(self):
        self.frozen = True
        self.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode=True):
        # Parent ``train()`` calls cannot accidentally reactivate frozen BN/dropout.
        return super().train(False if self.frozen else mode)

    @torch.no_grad()
    def clip_weights(self):
        weight = self.block2[0].weight
        weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=1.0))


class SpatialAttentionBlock(nn.Module):
    def __init__(self, attention, dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = attention
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * dim, dim))

    def forward(self, tokens):
        tokens = tokens + self.drop(self.attention(self.norm1(tokens)))
        return tokens + self.drop(self.ff(self.norm2(tokens)))


class FeatureClassifier(nn.Module):
    """Compare raw-feature and Tucker representations under identical masking.

    All arms receive the true electrode-availability vector. Fixed-size
    baseline tokens use explicitly flagged, learned missing placeholders; this
    is not claimed to implement an exact attention key mask. That convention
    keeps MHA and Performer comparable without attention-specific masking
    modifications.

    Core representations mix Rc latent components, not anatomical electrodes.
    Completion representations retain C electrode tokens. All attention is
    applied separately within each of the P windows.
    """

    REPRESENTATIONS = ('baseline', 'tensor_core', 'tensor_completion',
                       'linear_core', 'mean_completion')

    def __init__(self, attention, representation='baseline', *, channels=22,
                 windows=4, features=32, num_classes=4, dim=32, heads=4,
                 depth=1, dropout=.1, rank_channels=4, rank_features=4,
                 tensor=None, train_feature_mean=None, attention_options=None):
        super().__init__()
        _positive(channels=channels, windows=windows, features=features,
                  num_classes=num_classes, dim=dim, heads=heads, depth=depth,
                  rank_channels=rank_channels, rank_features=rank_features)
        if dim % heads or not 0 <= dropout < 1:
            raise ValueError('dim must divide into heads, and dropout must lie in [0,1)')
        if representation not in self.REPRESENTATIONS:
            raise ValueError(f'Unknown representation {representation!r}')
        if rank_channels > channels or rank_features > features:
            raise ValueError('Tucker ranks cannot exceed their input dimensions')
        self.attention_key, self.representation = attention, representation
        self.channels, self.windows, self.features = channels, windows, features
        self.rank_channels, self.rank_features = rank_channels, rank_features
        self.tensor = tensor
        if representation.startswith('tensor_'):
            if tensor is None:
                raise ValueError('Tensor representations require fitted training-only Tucker factors')
            expected_tensor_shape = (channels, features, rank_channels, rank_features)
            actual_tensor_shape = tuple(getattr(tensor, name, None) for name in
                                        ('channels', 'features', 'rank_channels', 'rank_features'))
            if actual_tensor_shape != expected_tensor_shape:
                raise ValueError('Tucker factor dimensions do not match the classifier representation')
            # Fitted factors remain shared/frozen; classifier gradients must never
            # fine-tune them with labels or contaminate another experiment arm.
            tensor.requires_grad_(False)
            tensor.eval()
        elif tensor is not None:
            raise ValueError('Supply tensor factors only to a tensor representation')
        self.is_core = representation in ('tensor_core', 'linear_core')
        self.num_tokens = rank_channels if self.is_core else channels
        self.token_axis = 'latent_component' if self.is_core else 'electrode'
        input_features = rank_features if self.is_core else features
        if representation == 'linear_core':
            # Shape-matched trainable compression; no tensor reconstruction loss.
            self.feature_bottleneck = nn.Linear(features, rank_features, bias=False)
            self.channel_bottleneck = nn.Linear(channels, rank_channels, bias=False)
        if representation == 'mean_completion':
            if train_feature_mean is None or tuple(train_feature_mean.shape) != (channels, features):
                raise ValueError('Mean completion needs the training-only mean [C,F]')
            mean = torch.as_tensor(train_feature_mean).detach().clone().float()
            if not torch.isfinite(mean).all():
                raise ValueError('Training feature means must be finite')
        else:
            mean = torch.zeros(channels, features)
        self.register_buffer('train_feature_mean', mean)
        self.input_projection = nn.Linear(input_features, dim)
        self.node_identity = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        # A node's observed flag (or core reliability fraction) and the complete
        # mask distinguish absent data from a measured numerical zero.
        self.status_embedding = nn.Linear(1, dim, bias=False)
        self.mask_embedding = nn.Linear(channels, dim, bias=False)
        self.missing_token = (nn.Parameter(torch.zeros(1, channels, dim))
                              if representation == 'baseline' else None)
        self.output_norm = nn.LayerNorm(dim)
        self.output_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)
        settings = dict(attention_options or {})
        if settings.get('heads', heads) != heads:
            raise ValueError('Attention heads must agree with the classifier')
        settings['heads'] = heads
        factory = AttentionFactory(attention, settings,
                                   modality=None if self.is_core else 'eeg',
                                   channels=channels)
        # Construct every non-attention parameter before allocating attention.
        # AttentionFactory isolates attention RNG draws across different methods.
        scaffold = [SpatialAttentionBlock(nn.Identity(), dim, dropout) for _ in range(depth)]
        self.blocks = nn.ModuleList(scaffold)
        for index, block in enumerate(self.blocks):
            block.attention = factory(dim, self.num_tokens, layer_idx=index,
                                      depth=depth, token_axis=self.token_axis)
        self.clip_weights()

    def _represent(self, observed, mask):
        if self.representation == 'tensor_core':
            return self.tensor.encode(observed, mask)
        if self.representation == 'tensor_completion':
            return self.tensor.complete(observed, mask)
        if self.representation == 'mean_completion':
            return torch.where(mask[..., None], observed,
                               self.train_feature_mean[None, :, None, :])
        if self.representation == 'linear_core':
            projected = self.feature_bottleneck(observed)  # B,C,P,Rf
            projected = self.channel_bottleneck(projected.permute(0, 2, 3, 1))
            return projected.permute(0, 3, 1, 2)  # B,Rc,P,Rf
        return observed

    def forward(self, features, mask=None):
        expected = (self.channels, self.windows, self.features)
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(f'Features must have shape [B,{expected[0]},{expected[1]},{expected[2]}]')
        batch = features.shape[0]
        mask = _availability(mask, batch=batch, channels=self.channels,
                             windows=self.windows, device=features.device)
        observed = torch.where(mask[..., None], features, torch.zeros_like(features))
        if not torch.isfinite(observed).all():
            raise ValueError('Observed features contain non-finite values')
        represented = self._represent(observed, mask)
        token_features = represented.permute(0, 2, 1, 3).reshape(
            batch * self.windows, self.num_tokens, -1)
        window_mask = mask.permute(0, 2, 1).reshape(batch * self.windows, self.channels)
        tokens = self.input_projection(token_features)
        if self.representation == 'baseline':
            tokens = torch.where(window_mask[..., None], tokens, self.missing_token)
        status = (window_mask.float().mean(-1, keepdim=True).expand(-1, self.num_tokens)
                  if self.is_core else window_mask.to(tokens.dtype))
        status = status.to(tokens.dtype)
        tokens = (tokens + self.node_identity + self.status_embedding(status[..., None])
                  + self.mask_embedding(window_mask.to(tokens.dtype))[:, None, :])
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        # Baseline readout counts only observed electrode tokens. Completion
        # readout uses measured+inferred electrodes, still carrying status flags.
        if self.representation == 'baseline':
            weight = window_mask.to(tokens.dtype)
        else:
            weight = torch.ones_like(status)
        per_window = (tokens * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1)
        per_window = per_window.reshape(batch, self.windows, -1)
        available_windows = mask.any(1).to(tokens.dtype)
        pooled = (per_window * available_windows[..., None]).sum(1)
        pooled = pooled / available_windows.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(self.output_drop(pooled))

    def train(self, mode=True):
        super().train(mode)
        if self.tensor is not None:
            self.tensor.eval()
        return self

    @torch.no_grad()
    def clip_weights(self):
        weight = self.classifier.weight
        weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=.25))


class EEGPretrainModel(nn.Module):
    """Supervised common-encoder pretraining with an MHA spatial head.

    Train only on training labels and select using validation. Discard the head
    afterward; freeze encoder once and reuse the exact features for every arm.
    """

    def __init__(self, encoder, *, num_classes=4, dim=32, heads=4, depth=1,
                 dropout=.1):
        super().__init__()
        self.encoder = encoder
        self.head = FeatureClassifier('mha', channels=encoder.channels,
                                      windows=encoder.windows, features=encoder.features,
                                      num_classes=num_classes, dim=dim, heads=heads,
                                      depth=depth, dropout=dropout)

    def forward(self, raw, mask=None):
        features = self.encoder(raw, mask)
        return self.head(features, mask)

    def clip_weights(self):
        self.encoder.clip_weights()
        self.head.clip_weights()
