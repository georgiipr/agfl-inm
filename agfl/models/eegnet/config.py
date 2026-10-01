EEG_DEFAULTS = {
    'temp_kernel': 32, 'f1': 16, 'd': 2, 'f2': 32, 'pk1': 8, 'pk2': 16,
    'dropout_rate': .5, 'max_norm1': 1., 'max_norm2': .25, 'attention_axis': 'electrode',
    'spatial_readout': 'learned', 'attention_residual': True,
    'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5,
    'attention_dropout': 0.0, 'initialization': 'pytorch',
    'electrode_dim': 32, 'electrode_architecture': 'spatial_fusion',
}
