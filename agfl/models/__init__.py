"""EEG backbones; attention is a separate model-construction argument."""
from importlib import import_module

MODEL_KEYS = ('eegnet', 'signal_transformer')


def model_registry():
    return {key: import_module(f'{__name__}.{key}').SPEC for key in MODEL_KEYS}


def get_model_spec(key):
    if key not in MODEL_KEYS:
        raise ValueError(f'Unknown model {key!r}; choose from {MODEL_KEYS}. Select attention separately.')
    return model_registry()[key]


def available_models():
    return list(MODEL_KEYS)
