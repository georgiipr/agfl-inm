"""ATCNet's fixed MHA equations for cluster TensorFlow versions without MHA.

This compatibility layer supports only the checkpoint's rank-three, unmasked
two-head self-attention. It retains the eight Keras projection tensors in their
original order and shape; unsupported options fail. Native Keras MHA is used
when present. Cluster preflight compares the fallback against independent NumPy
equations, checkpoint tensor bytes, and native MHA when available.

Equation/weight-layout reference:
https://github.com/keras-team/keras/blob/v2.15.0/keras/layers/attention/multi_head_attention.py
"""
import math
import tensorflow as tf


class ATCNetMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        if num_heads != 2 or key_dim != 8:
            raise ValueError('The compatibility layer supports only ATCNet heads=2, key_dim=8')
        super().__init__(**kwargs)
        self.num_heads, self.key_dim, self.dropout = num_heads, key_dim, dropout
        self._attention_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        shape = tf.TensorShape(input_shape)
        if shape.rank != 3 or shape[-1] != 32:
            raise ValueError('ATCNet MHA expects [batch, time, 32]')
        # Topological H5 loading uses this exact variable order: Q, K, V, output.
        for projection in ('query', 'key', 'value'):
            setattr(self, projection + '_kernel', self.add_weight(
                name=projection + '/kernel', shape=(32, 2, 8), initializer='glorot_uniform'))
            setattr(self, projection + '_bias', self.add_weight(
                name=projection + '/bias', shape=(2, 8), initializer='zeros'))
        self.output_kernel = self.add_weight(
            name='attention_output/kernel', shape=(2, 8, 32), initializer='glorot_uniform')
        self.output_bias = self.add_weight(
            name='attention_output/bias', shape=(32,), initializer='zeros')
        super().build(input_shape)

    def call(self, query, value, key=None, training=None,
             return_attention_scores=False, attention_mask=None, use_causal_mask=False):
        if attention_mask is not None or use_causal_mask:
            raise ValueError('The published ATCNet checkpoint uses unmasked temporal attention')
        if key is None:
            key = value
        q = tf.einsum('abc,cde->abde', query, self.query_kernel) + self.query_bias
        k = tf.einsum('abc,cde->abde', key, self.key_kernel) + self.key_bias
        v = tf.einsum('abc,cde->abde', value, self.value_kernel) + self.value_bias
        q = tf.multiply(q, 1.0 / math.sqrt(float(self.key_dim)))
        scores = tf.nn.softmax(tf.einsum('aecd,abcd->acbe', k, q), axis=-1)
        dropped = self._attention_dropout(scores, training=training)
        context = tf.einsum('acbe,aecd->abcd', dropped, v)
        output = tf.einsum('abcd,cde->abe', context, self.output_kernel) + self.output_bias
        return (output, scores) if return_attention_scores else output

    def get_config(self):
        return dict(super().get_config(), num_heads=self.num_heads,
                    key_dim=self.key_dim, dropout=self.dropout)


NATIVE_MHA = getattr(tf.keras.layers, 'MultiHeadAttention', None)
MultiHeadAttention = NATIVE_MHA or ATCNetMultiHeadAttention


def install_upstream_compatibility():
    """Supply only the missing API before importing the immutable author source.

    This is explicit runtime adaptation: the upstream files remain hash-verified.
    The native-reference receipt records when the shared fallback was required;
    wrapper/upstream parity alone then cannot independently validate MHA.
    """
    import importlib
    # Old Keras exports l2 as a factory rather than the separate L2 class.
    # Frozen-model regularizer metadata is retained; no kernel loss is trained.
    for name in ('tensorflow.keras.regularizers', 'tensorflow.python.keras.regularizers'):
        regularizers = importlib.import_module(name)
        if not hasattr(regularizers, 'L2'):
            regularizers.L2 = regularizers.l2
    if NATIVE_MHA is None:
        tf.keras.layers.MultiHeadAttention = ATCNetMultiHeadAttention
        for name in ('tensorflow.keras.layers', 'tensorflow.python.keras.layers'):
            module = importlib.import_module(name)
            if not hasattr(module, 'MultiHeadAttention'):
                setattr(module, 'MultiHeadAttention', ATCNetMultiHeadAttention)


def compatibility_record():
    return {'tensorflow': tf.__version__,
            'attention_implementation': 'native_keras' if NATIVE_MHA else 'atcnet_legacy_compat',
            'upstream_reference_shares_legacy_attention': NATIVE_MHA is None}
