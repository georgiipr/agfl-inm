"""EEG model construction; the experiment workflow is owned by inm.study."""
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class ModelSpec:
    key: str
    defaults: dict
    constructor: object

    def defaults_for(self, modality='eeg'):
        if modality != 'eeg':
            raise ValueError('Only EEG models are supported')
        return deepcopy(self.defaults)

    def build(self, model_options, metadata, attention='mha', attention_options=None):
        from agfl.attention import AttentionFactory, get_attention_spec
        from agfl.config import merge
        from .modality import validate_attention_domain, validate_model_graph

        defaults = self.defaults_for(metadata['modality'])
        unknown = set(model_options) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown {self.key} model_options: {sorted(unknown)}')
        options = merge(defaults, model_options)
        settings = merge(get_attention_spec(attention).defaults, attention_options or {})
        validate_attention_domain(metadata['modality'], options, settings)
        factory = AttentionFactory(attention, settings, modality='eeg', channels=metadata['channels'])
        model = self.constructor(options, metadata, factory)
        validate_model_graph(model, metadata)
        return model
