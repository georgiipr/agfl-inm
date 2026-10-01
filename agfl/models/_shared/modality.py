"""EEG attention mixes electrode identities, independently of temporal encoding."""


def validate_attention_domain(modality, model_options, attention_options):
    if modality != 'eeg':
        raise ValueError('Only EEG models are supported')
    if model_options.get('attention_axis', 'electrode') != 'electrode':
        raise ValueError('EEG attention must operate across electrodes')
    if 'electrode_dim' in model_options:
        dim = model_options['electrode_dim']
        heads = attention_options.get('heads', 4)
        if type(dim) is not int or dim < 1 or type(heads) is not int or heads < 1 or dim % heads:
            raise ValueError('EEGNet electrode_dim must be positive and divisible by attention heads')


def validate_model_graph(model, metadata):
    layers = [m for m in model.modules() if getattr(m, 'is_attention', False)]
    if not layers:
        raise ValueError('A model must contain the selected attention mechanism')
    for layer in [model, *layers]:
        if getattr(layer, 'token_axis', None) != 'electrode' or layer.num_tokens != metadata['channels']:
            raise ValueError('Every EEG attention layer must have one node per input electrode')
