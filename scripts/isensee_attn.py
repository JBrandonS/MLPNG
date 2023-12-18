import os

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from config import SimConfig
from dataloader import DataLoader
from imagesavinglayer import ImageSavingLayer

from keras import backend as K
from matplotlib import pyplot as plt
from sklearn import metrics as mt
from tensorflow import pad
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import (
    Activation,
    Add,
    Attention,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GroupNormalization,
    Input,
    Lambda,
    Layer,
    LayerNormalization,
    LeakyReLU,
    MaxPooling1D,
    MultiHeadAttention,
    Multiply,
    PReLU,
    SeparableConv2D,
    SpatialDropout2D,
    Subtract,
    UpSampling2D,
    UnitNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import l2

@tf.function
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.function
def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "padding": self.padding,
                "input_spec": self.input_spec,
            }
        )
        return config

def create_localization_module(input_layer, n_filters):
    """
    This function creates a localization module, which is typically used in the
    decoder part of a U-Net architecture. It first applies reflection padding to
    the input, then applies two convolutional blocks. The second convolutional block
    uses a 1x1 kernel, which is often used to map each
    64-component feature vector to the desired number of classes.
    """
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    return create_convolution_block(convolution1, n_filters, kernel=(1, 1))

def create_up_sampling_module(
    input_layer, n_filters, size=(2, 2), interpolation="nearest"
):
    """
    This function creates an up-sampling module, which is used to increase the spatial
    dimensions of the input. It first applies up-sampling to the input using nearest
    neighbor interpolation, then applies reflection padding and a convolutional block.
    """
    up_sample = UpSampling2D(size=size, interpolation=interpolation)(input_layer)
    layer1 = ReflectionPadding2D()(up_sample)
    return create_convolution_block(layer1, n_filters)

def create_context_module(
    input_layer, n_level_filters, dropout_rate=0.1, data_format="channels_last"
):
    """
    This function creates a context module, which is typically used in the encoder part
    of a U-Net architecture. It first applies reflection padding to the input, then applies
    a convolutional block, a spatial dropout layer (which randomly sets entire feature maps
    to zero), another reflection padding layer, and another convolutional block.
    """
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(
        input_layer=layer1, n_filters=n_level_filters
    )
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = ReflectionPadding2D()(dropout)
    return create_convolution_block(input_layer=layer2, n_filters=n_level_filters)

def create_convolution_block(
    input_layer,
    n_filters,
    batch_normalization=False,
    kernel=(3, 3),
    activation="LeakyReLU", # "relu"
    kernel_initializer="he_uniform",
    padding="valid",
    strides=(1, 1),
    instance_normalization=True,
    kernel_regularizer=None,
):
    """
    This function creates a convolutional block, which is a fundamental building block of CNNs.
    It applies a 2D convolution to the input, followed by either batch normalization or instance
    normalization (if enabled). The number of filters, kernel size, activation function, padding,
    strides, and kernel initializer for the convolution can be specified as parameters.
    """
    layer = Conv2D(
        n_filters,
        kernel,
        padding=padding,
        strides=strides,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        # activation=activation,
    )(input_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        layer = GroupNormalization(groups=n_filters)(layer)
    layer = Activation(activation=activation)(layer)
    return layer

def isensee_attn(
    inputs,
    n_base_filters=16,
    depth=5,
    dropout_rate=0.3,
    n_segmentation_levels=3,
    n_labels=8,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="",
    metrics=[],
    interpolation="bilinear",
    kernel_regularizer=None,
    attn_heads=2,
    attn_key_dim=64,
):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param inputs:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """

    current_layer = inputs
    level_output_layers = []
    level_filters = []
    n_level_filters = n_base_filters
    for _ in range(depth):
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(
                layer,
                n_level_filters,
                strides=(2, 2),
                kernel_regularizer=kernel_regularizer,
            )

        # Self-attention
        # attention = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(
        #     in_conv, in_conv
        # )
        # attention_output = Multiply()([in_conv, attention])

        context_output_layer = create_context_module(
            in_conv,
            # attention_output,
            n_level_filters,
            dropout_rate=dropout_rate,  # in_cov -> attention
        )

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = []
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number], interpolation=interpolation
        )

        # Reg attention
        attention = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(
            level_output_layers[level_number], up_sampling
        )
        comb_attention = LayerNormalization(epsilon=1e-6)(up_sampling + attention)

        concatenation_layer = Concatenate()(
            [
                level_output_layers[level_number],
                comb_attention,
                # up_sampling,
            ]  # up_sampling -> comb_attention
        )
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number]
        )
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(
                0,
                Conv2D(
                    n_labels,
                    (1, 1),
                )(current_layer),
            )

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels - 1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2), interpolation=interpolation)(
                output_layer
            )

    out_layer = Flatten()(output_layer)
    # out_layer = Dropout(dropout_rate)(out_layer)
    # out_layer = Dense(
    #     1024,
    #     activation="sigmoid",
    #     kernel_initializer="glorot_uniform",
    #     kernel_regularizer=kernel_regularizer,
    # )(out_layer)
    out_layer = Dense(1)(out_layer)

    model = Model(inputs=inputs, outputs=out_layer, name=name)

    # Allows for us to pass in a complete optimizer or incomplete with learning rate
    if callable(optimizer):
        optimizer = optimizer(learning_rate=initial_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model
