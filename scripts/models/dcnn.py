import math
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Attention,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    Multiply,
    SpatialDropout2D,
)

from scripts.models import ModelCore, register_model
from scripts.utils.tf.layers import ReflectionPadding2D, augmentation_layer


@register_model
class DCNN(ModelCore):
    def __init__(self, argv=None):
        super().__init__(argv)
        self.BATCH_SIZE = 4

    def deep_cnn_block(
        self,
        n_filters,
        depth=2,
        kernel=(3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="valid",
        strides=(1, 1),
        kernel_regularizer=None,
        dropuot_rate=0.3,
    ):
        pad = (kernel[0] - 1) // 2

        # create a CNN block
        layers = []
        for _ in range(depth):
            layers.append(
                Conv2D(
                    n_filters,
                    kernel,
                    padding=padding,
                    strides=strides,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activation=activation,
                )
            )
            layers.append(ReflectionPadding2D((pad, pad)))
            layers.append(Conv2D(n_filters, kernel, padding=padding, strides=strides))
            layers.append(ReflectionPadding2D((pad, pad)))
            layers.append(BatchNormalization())
            layers.append(Dropout(dropuot_rate))

        return tf.keras.Sequential(layers)

    def _model(
        self,
        inputs,
        depth=5,
        dropout_rate=0.3,
        kernel_regularizer=None,
        flip=True,
        rotate=True,
        add_powers=2,
    ):
        # this finds the number of factor of 2 reductions in the spatial dimensions to make final depth 16x16
        # just to ensure we don't go too deep / small
        max_depth = math.log(inputs.shape[1] / 8, 2) + 1
        depth = min(depth, int(max_depth))

        layer = augmentation_layer(flip, rotate, add_powers)(inputs)
        for level in range(depth):
            n_level_filters = 2 ** (5 + level)
            res_layer = layer

            layer = self.deep_cnn_block(
                n_level_filters, kernel_regularizer=kernel_regularizer
            )(layer)

            # add attention and residual connection
            res_layer = Conv2D(n_level_filters, (1, 1))(res_layer)
            att_layer = Attention(use_scale=True)([res_layer, layer])
            layer = Multiply()([att_layer, layer])
            layer = Add()([layer, res_layer])

            # downsample
            layer = MaxPooling2D((2, 2))(layer)
            layer = SpatialDropout2D(dropout_rate)(layer)

        out_layer = Flatten()(layer)
        return Dense(1)(out_layer)
