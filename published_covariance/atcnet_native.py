"""Minimal native EEG-ATCNet definitions used by the frozen wrapper.

Copyright (C) 2022 King Saud University, Saudi Arabia
SPDX-License-Identifier: Apache-2.0

These definitions are reproduced from Hamdi Altaheri's EEG-ATCNet repository,
revision 65162fb359ea46a2f62c885a9987247ab491ee7a.  They retain the original
TensorFlow/Keras operations and layer defaults for ATCNet_(attention='mha').
The complete license is distributed with the audited source asset at
https://github.com/Altaheri/EEG-ATCNet/blob/65162fb359ea46a2f62c885a9987247ab491ee7a/LICENSE.
This module omits unrelated architectures and unsupported attention variants.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Input,
    LayerNormalization,
    Lambda,
    MultiHeadAttention,
    Permute,
)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm


def attention_block(in_layer, attention_model):
    """Pinned source's MHA branch from attention_models.attention_block."""
    # The model's input is rank three, so the source branch does not reshape.
    x = LayerNormalization(epsilon=1e-6)(in_layer)
    x = MultiHeadAttention(key_dim=8, num_heads=2, dropout=0.5)(x, x)
    x = Dropout(0.3)(x)
    return Add()([in_layer, x])


def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2,
                in_chans=22, weightDecay=0.009, maxNorm=0.6, dropout=0.25):
    F2 = F1 * D
    block1 = Conv2D(
        F1, (kernLength, 1), padding="same", data_format="channels_last",
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False,
    )(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D(
        (1, in_chans), depth_multiplier=D, data_format="channels_last",
        depthwise_regularizer=L2(weightDecay),
        depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False,
    )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((8, 1), data_format="channels_last")(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(
        F2, (16, 1), data_format="channels_last",
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False,
        padding="same",
    )(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation("elu")(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format="channels_last")(block3)
    return Dropout(dropout)(block3)


def TCN_block_(input_layer, input_dimension, depth, kernel_size, filters,
               dropout, weightDecay=0.009, maxNorm=0.6, activation="relu"):
    block = Conv1D(
        filters, kernel_size=kernel_size, dilation_rate=1, activation="linear",
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1]), padding="causal",
        kernel_initializer="he_uniform",
    )(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(
        filters, kernel_size=kernel_size, dilation_rate=1, activation="linear",
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNorm, axis=[0, 1]), padding="causal",
        kernel_initializer="he_uniform",
    )(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv = Conv1D(
            filters, kernel_size=1, kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNorm, axis=[0, 1]), padding="same",
        )(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)
    for i in range(depth - 1):
        block = Conv1D(
            filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
            activation="linear", kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNorm, axis=[0, 1]), padding="causal",
            kernel_initializer="he_uniform",
        )(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(
            filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
            activation="linear", kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNorm, axis=[0, 1]), padding="causal",
            kernel_initializer="he_uniform",
        )(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out


def ATCNet_(n_classes, in_chans=22, in_samples=1125, n_windows=5,
            attention="mha", eegn_F1=16, eegn_D=2, eegn_kernelSize=64,
            eegn_poolSize=7, eegn_dropout=0.3, tcn_depth=2,
            tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation="elu", fuse="average"):
    """Native ATCNet_ defaults and graph from pinned ``models.py``."""
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)
    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    block1 = Conv_block_(
        input_layer=input_2, F1=eegn_F1, D=eegn_D,
        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
        weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
        in_chans=in_chans, dropout=eegn_dropout,
    )
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
    sw_concat = []
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]
        if attention is not None:
            block2 = attention_block(block2, attention)
        block3 = TCN_block_(
            input_layer=block2, input_dimension=eegn_F1 * eegn_D,
            depth=tcn_depth, kernel_size=tcn_kernelSize, filters=tcn_filters,
            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
            dropout=tcn_dropout, activation=tcn_activation,
        )
        block3 = Lambda(lambda x: x[:, -1, :])(block3)
        if fuse == "average":
            sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))
        elif fuse == "concat":
            raise NotImplementedError("The published frozen checkpoint uses fuse='average'.")
        else:
            raise ValueError(f"unsupported fusion mode: {fuse!r}")
    if fuse == "average":
        sw_concat = tf.keras.layers.Average()(sw_concat) if len(sw_concat) > 1 else sw_concat[0]
    out = Activation("softmax", name="softmax")(sw_concat)
    return Model(inputs=input_1, outputs=out)
