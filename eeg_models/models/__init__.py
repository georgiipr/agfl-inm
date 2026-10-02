"""Four concrete models. Attention is built in, never selected or injected."""
from eeg_models.config import merge

MODEL_KEYS = ('eegnet', 'signal_transformer', 'eegnet_tensor', 'signal_transformer_tensor')
MODEL_LABELS = {
    'eegnet': 'EEGNet', 'signal_transformer': 'Signal Transformer',
    'eegnet_tensor': 'EEGNet + tensors', 'signal_transformer_tensor': 'Signal Transformer + tensors',
}


def model_definition(key):
    if key not in MODEL_KEYS:
        raise ValueError(f'Unknown model {key!r}; choose from {MODEL_KEYS}')
    backbone = key.removesuffix('_tensor')
    return {'name': key, 'backbone': backbone, 'tensor': key.endswith('_tensor')}


def build_model(key, metadata, model_options=None, *, window_samples=250, tensor_options=None):
    definition = model_definition(key)
    if metadata.get('modality') != 'eeg':
        raise ValueError('Models require EEG metadata')
    if definition['backbone'] == 'eegnet':
        from .eegnet.config import EEG_DEFAULTS as defaults
        from .eegnet import EEGNet, EEGNetTensor
        constructor = EEGNetTensor if definition['tensor'] else EEGNet
    else:
        from .signal_transformer.config import DEFAULTS as defaults
        from .signal_transformer import SignalTransformer, SignalTransformerTensor
        constructor = SignalTransformerTensor if definition['tensor'] else SignalTransformer
    model_options = model_options or {}
    unknown = set(model_options) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown {key} model options: {sorted(unknown)}')
    options = merge(defaults, model_options)
    arguments = {'window_samples': window_samples}
    if definition['tensor']:
        arguments['tensor_options'] = tensor_options
    return constructor(options, metadata, **arguments)
