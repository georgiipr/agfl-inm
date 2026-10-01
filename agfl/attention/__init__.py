"""MHA and Performer, selected independently of the EEG backbone."""
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module

ATTENTION_KEYS = ('mha', 'performer')


@dataclass(frozen=True)
class AttentionSpec:
    key: str
    defaults: dict
    constructor: object

    def build(self, options, dim, tokens):
        from agfl.config import merge
        unknown = set(options) - set(self.defaults)
        if unknown:
            raise ValueError(f'Unknown {self.key} attention_options: {sorted(unknown)}')
        settings = {**merge(self.defaults, options), 'dim': dim}
        heads = settings['heads']
        if type(heads) is not int or heads < 1 or dim % heads:
            raise ValueError(f'attention_options.heads={heads} must divide the model feature dimension {dim}')
        return self.constructor(settings, tokens)


def attention_registry():
    # Explicit names prevent stale source folders on a cluster from reappearing.
    return {key: import_module(f'{__name__}.{key}').SPEC for key in ATTENTION_KEYS}


def get_attention_spec(key):
    registry = attention_registry()
    if key not in registry:
        raise ValueError(f'Unknown attention {key!r}; available: {", ".join(registry)}')
    return registry[key]


def available_attentions():
    return sorted(attention_registry())


class AttentionFactory:
    """Inject attention without shifting initialization of the surrounding model."""
    def __init__(self, key, options, *, modality=None, channels=None):
        self.spec, self.options = get_attention_spec(key), deepcopy(options)
        self.count = 0
        if modality not in (None, 'eeg'):
            raise ValueError('Only EEG spatial attention is supported')
        self.modality, self.channels = modality, channels

    def __call__(self, dim, tokens, layer_idx=0, depth=1, token_axis='electrode'):
        import torch
        if token_axis not in ('electrode', 'latent_component'):
            raise ValueError('Attention tokens must be electrodes or latent spatial components')
        if self.modality == 'eeg' and (token_axis != 'electrode' or tokens != self.channels):
            raise ValueError('EEG attention requires one graph node per electrode')
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator().manual_seed((torch.initial_seed() + 104729 * (self.count + 1)) % (2**63))
            torch.set_rng_state(generator.get_state())
            module = self.spec.build(self.options, dim, tokens)
        self.count += 1
        module.attention_key = self.spec.key
        module.is_attention = True
        module.num_tokens = tokens
        module.token_axis = token_axis
        module.layer_idx, module.depth = layer_idx, depth
        return module
