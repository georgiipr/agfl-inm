"""Cluster-only admission for runtime, checkpoint mapping and legacy MHA math."""
import json
import numpy as np
import tensorflow as tf
from .classifier import FrozenClassifier, _verified_checkpoint
from .tf_compat import ATCNetMultiHeadAttention, NATIVE_MHA, compatibility_record


def _numpy_attention(x, weights):
    qk, qb, kk, kb, vk, vb, ok, ob = weights
    q = np.einsum('btc,chd->bthd', x, qk) + qb
    k = np.einsum('bsc,chd->bshd', x, kk) + kb
    v = np.einsum('bsc,chd->bshd', x, vk) + vb
    scores = np.einsum('bthd,bshd->bhts', q / np.sqrt(np.float32(8)), k)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores /= scores.sum(axis=-1, keepdims=True)
    context = np.einsum('bhts,bshd->bthd', scores, v)
    return np.einsum('bthd,hdc->btc', context, ok) + ob


def _checkpoint_arrays(path):
    # h5py is already a TensorFlow runtime dependency; no separate install.
    import h5py
    with h5py.File(str(path), 'r') as handle:
        root = handle['model_weights'] if 'model_weights' in handle else handle
        result = []
        for name in root.attrs['layer_names']:
            if isinstance(name, bytes):
                name = name.decode('utf8')
            layer = root[name]
            names = layer.attrs.get('weight_names', ())
            if len(names):
                result.append([np.asarray(layer[n.decode('utf8') if isinstance(n, bytes) else n]) for n in names])
        return result


def validate_runtime_and_attention(assets):
    if not tf.executing_eagerly() or not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Published reproduction requires eager TensorFlow and a visible GPU')
    if not hasattr(np.random, 'PCG64'):
        raise RuntimeError('The existing module needs NumPy >=1.17 for the declared PCG64 mask protocol')
    rng = np.random.Generator(np.random.PCG64(20261007))
    values = rng.normal(size=(2, 7, 32)).astype(np.float32)
    checked = []
    for i in range(1, 10):
        subject = f'A{i:02d}'
        tf.random.set_seed(20261007)
        with tf.device('/GPU:0'):
            classifier = FrozenClassifier(subject, assets_root=assets)
        saved = _checkpoint_arrays(_verified_checkpoint(subject, assets))
        loaded = [layer.get_weights() for layer in classifier.model.layers if layer.weights]
        if len(saved) != len(loaded):
            raise RuntimeError(subject + ' checkpoint layer inventory differs')
        for expected, actual in zip(saved, loaded):
            if len(expected) != len(actual) or any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
                raise RuntimeError(subject + ' H5 projection/normalization tensor mapping differs')
        attention = [layer for layer in classifier.model.layers
                     if isinstance(layer, ATCNetMultiHeadAttention)
                     or (NATIVE_MHA is not None and isinstance(layer, NATIVE_MHA))]
        if len(attention) != 5:
            raise RuntimeError(subject + ' must contain five published temporal MHA branches')
        for layer in attention:
            weights = layer.get_weights()
            expected_shapes = [(32, 2, 8), (2, 8)] * 3 + [(2, 8, 32), (32,)]
            if [a.shape for a in weights] != expected_shapes:
                raise RuntimeError('Published MHA tensor order/shapes differ')
            with tf.device('/GPU:0'):
                fallback = ATCNetMultiHeadAttention(num_heads=2, key_dim=8, dropout=0.5)
                fallback(values, values, training=False)
                fallback.set_weights(weights)
                x = tf.Variable(values)
                with tf.GradientTape() as tape:
                    actual = fallback(x, x, training=False)
                    loss = tf.reduce_sum(tf.square(actual))
                gradient = tape.gradient(loss, x)
            expected = _numpy_attention(values, weights)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-4, atol=2e-5)
            if 'GPU:0' not in actual.device or gradient is None or not np.isfinite(gradient.numpy()).all():
                raise RuntimeError('Legacy MHA GPU output/input-gradient admission failed')
            if NATIVE_MHA is not None:
                np.testing.assert_allclose(actual.numpy(), layer(values, values, training=False).numpy(),
                                           rtol=2e-4, atol=2e-5)
        checked.append(subject)
        del classifier
        tf.keras.backend.clear_session()
    print(json.dumps({'runtime_preflight': 'passed', 'checkpoint_subjects': checked,
                      'tensor_bytes_checked': True, 'numpy_attention_parity': True,
                      **compatibility_record()}, sort_keys=True), flush=True)
