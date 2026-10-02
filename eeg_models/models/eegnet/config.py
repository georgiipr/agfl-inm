"""Full-trial EEGNet with MHA before the depthwise spatial filter."""
EEG_DEFAULTS = {
    'temp_kernel': 125, 'f1': 16, 'd': 2, 'f2': 32,
    'pk1': 8, 'pk2': 16, 'dropout_rate': 0.5,
    'max_norm1': 1.0, 'max_norm2': 0.25,
    'batch_norm_momentum': 0.1, 'batch_norm_eps': 1e-5,
}
