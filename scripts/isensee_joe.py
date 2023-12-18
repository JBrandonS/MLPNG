import os
import tensorflow as tf
from tensorflow.keras import backend as K
import sklearn.model_selection as model_selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import Input
from tensorflow.keras.layers import Attention, Layer, LeakyReLU, Add, UpSampling2D, Activation, SpatialDropout2D, Conv2D, BatchNormalization, Concatenate, Flatten, Dense, GroupNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow_addons.layers import InstanceNormalization

from tensorflow import pad
from functools import partial


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        # self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


def create_localization_module(input_layer, n_filters):
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2)):
    up_sample = UpSampling2D(size=size)(input_layer)
    layer1 = ReflectionPadding2D()(up_sample)
    convolution = create_convolution_block(layer1, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(input_layer=layer1, n_filters=n_level_filters)
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = ReflectionPadding2D()(dropout)
    convolution2 = create_convolution_block(input_layer=layer2, n_filters=n_level_filters)
    return convolution2


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                             padding='valid', strides=(1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        # layer = InstanceNormalization()(layer)
        layer = GroupNormalization(groups=n_filters)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

def isensee2017_joe(inputs, n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=dice_coefficient_loss, activation_name="relu"):
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
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
#         n_level_filters = (2**level_number) * n_base_filters
        n_level_filters = 16
        #n_level_filters = 32
        #n_level_filters = 64
        #n_level_filters = 48
        level_filters.append(n_level_filters)
        

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters, strides=(2, 2))
            #in_conv = create_convolution_block(layer, n_level_filters, strides=(3, 3))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        
        #attention_layer = Attention(use_scale=True, dropout=dropout_rate)([level_output_layers[level_number], up_sampling])
        
        #concatenation_layer = Concatenate()([level_output_layers[level_number], up_sampling, attention_layer])
        
        concatenation_layer = Concatenate()([level_output_layers[level_number], up_sampling])
        
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        
        current_layer = localization_output
        
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels-1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2))(output_layer)
            #output_layer = UpSampling2D(size=(3, 3))(output_layer)

    flat_layer = Flatten()(output_layer)
    
    out_layer = Dense(1,activation=None)(flat_layer)
    
    # model = Model(inputs=inputs, outputs=activation_block)
    model = Model(inputs=inputs, outputs=out_layer)
    model.compile(optimizer=optimizer(learning_rate=initial_learning_rate), loss=loss_function, metrics=tf.keras.metrics.RootMeanSquaredError())
    return model