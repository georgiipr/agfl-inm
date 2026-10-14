"""Two backbones, each alone or with frozen/learned covariance; fixed MHA."""
from eeg_models.config import merge

MODEL_KEYS = ('eegnet', 'eegnet_covariance_frozen', 'eegnet_covariance_learned',
              'signal_transformer', 'signal_transformer_covariance_frozen', 'signal_transformer_covariance_learned')
MODEL_LABELS = {
    'eegnet': 'EEGNet', 'signal_transformer': 'Signal Transformer',
    'eegnet_covariance_frozen': 'EEGNet + frozen covariance',
    'eegnet_covariance_learned': 'EEGNet + learned covariance',
    'signal_transformer_covariance_frozen': 'Signal Transformer + frozen covariance',
    'signal_transformer_covariance_learned': 'Signal Transformer + learned covariance',
}


def model_definition(key):
    if key not in MODEL_KEYS:
        raise ValueError(f'Unknown model {key!r}; choose from {MODEL_KEYS}')
    backbone, separator, suffix = key.partition('_covariance_')
    return {'name': key, 'backbone': backbone,
            'completion': 'covariance_' + suffix if separator else 'none'}


def build_model(key, metadata, model_options=None, *, window_samples=250, covariance_options=None):
    definition = model_definition(key)
    if metadata.get('modality') != 'eeg':
        raise ValueError('Models require EEG metadata')
    if definition['backbone'] == 'eegnet':
        from .eegnet.config import EEG_DEFAULTS as defaults
        from .eegnet import EEGNet, EEGNetFrozenCovariance, EEGNetLearnedCovariance
        constructor = {'none': EEGNet, 'covariance_frozen': EEGNetFrozenCovariance,
                       'covariance_learned': EEGNetLearnedCovariance}[definition['completion']]
    else:
        from .signal_transformer.config import DEFAULTS as defaults
        from .signal_transformer import (SignalTransformer, SignalTransformerFrozenCovariance,
                                         SignalTransformerLearnedCovariance)
        constructor = {'none': SignalTransformer, 'covariance_frozen': SignalTransformerFrozenCovariance,
                       'covariance_learned': SignalTransformerLearnedCovariance}[definition['completion']]
    model_options = model_options or {}
    unknown = set(model_options) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown {key} model options: {sorted(unknown)}')
    options = merge(defaults, model_options)
    arguments = {'window_samples': window_samples}
    if definition['completion'] != 'none':
        arguments['covariance_options'] = covariance_options
    return constructor(options, metadata, **arguments)
