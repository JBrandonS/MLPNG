import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (
    Layer,
    LeakyReLU,
    ReLU,
    Add,
    UpSampling2D,
    Activation,
    SpatialDropout2D,
    Conv2D,
    BatchNormalization,
    Concatenate,
    Flatten,
    Dense,
    GroupNormalization,
)

from scripts.models import ModelCore, register_model
from scripts.utils.tf.dataloaders import PatchLoader
from scripts.utils.tf.dataloaders_v2 import DataLoaderBase_v2


class PeriodicPadding2D(Layer):
    def __init__(self, current_grid, **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.current_grid = current_grid
        self.indices = np.append(
            np.insert(np.arange(self.current_grid), 0, self.current_grid - 1), 0
        ).astype(np.int32)

    def call(self, x):
        x = tf.gather(x, self.indices, axis=1)
        x = tf.gather(x, self.indices, axis=2)
        return x


@register_model
class ISENSEE_V2(ModelCore):
    """This is a clone of model taken from Thomas' UNET_fnl notebook, with modifications"""

    def init_dataset(self, *args, **kwargs):
        self._dataset = PatchLoader(self.patch_file, *args, **kwargs)
        return self._dataset

    def create_localization_module(self, input_layer, current_grid, n_filters):
        layer1 = PeriodicPadding2D(current_grid)(input_layer)
        convolution1 = self.create_convolution_block(layer1, n_filters)
        convolution2 = self.create_convolution_block(
            convolution1, n_filters, kernel=(1, 1)
        )
        return convolution2

    def create_up_sampling_module(
        self, input_layer, current_grid, n_filters, size=(2, 2)
    ):
        # interpolation is set to bilinear because the default, nearest, is not XLA compatible
        up_sample = UpSampling2D(size=size, interpolation="bicubic")(input_layer)
        layer1 = PeriodicPadding2D(current_grid)(up_sample)
        convolution = self.create_convolution_block(layer1, n_filters)
        return convolution

    def create_context_module(
        self,
        input_layer,
        current_grid,
        n_level_filters,
        dropout_rate=0.3,
        data_format="channels_last",
    ):
        layer1 = PeriodicPadding2D(current_grid)(input_layer)
        convolution1 = self.create_convolution_block(
            input_layer=layer1, n_filters=n_level_filters
        )
        dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(
            convolution1
        )
        layer2 = PeriodicPadding2D(current_grid)(dropout)
        convolution2 = self.create_convolution_block(
            input_layer=layer2, n_filters=n_level_filters
        )
        return convolution2

    def create_convolution_block(
        self,
        input_layer,
        n_filters,
        batch_normalization=False,
        kernel=(3, 3),
        # activation=LeakyReLU,
        activation=ReLU,
        padding="valid",
        strides=(1, 1),
        instance_normalization=True,
    ):

        layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        elif instance_normalization:
            layer = GroupNormalization(n_filters)(layer)
        if activation is None:
            return Activation("relu")(layer)
        else:
            return activation()(layer)

    def _model(self, inputs, depth=7, n_base_filters=16, dropout_rate=0.3, n_labels=8):
        x = inputs
        level_output_layers = list()
        level_filters = list()
        current_grid = x.shape[1]
        print(f"Current grid: {current_grid}, {x.shape}")

        for level_number in range(depth):
            n_level_filters = (2**level_number) * n_base_filters
            level_filters.append(n_level_filters)

            if x is inputs:
                x = PeriodicPadding2D(current_grid)(x)
                x = self.create_convolution_block(x, n_level_filters)
            else:
                x = PeriodicPadding2D(current_grid)(x)
                x = self.create_convolution_block(x, n_level_filters, strides=(2, 2))
                current_grid /= 2

            previous_block = x
            x = self.create_context_module(
                x, current_grid, n_level_filters, dropout_rate=dropout_rate
            )
            x = Add()([previous_block, x])

            level_output_layers.append(x)

        for level_number in range(depth - 2, -1, -1):
            current_grid *= 2
            x = self.create_up_sampling_module(
                x, current_grid, level_filters[level_number]
            )
            x = Concatenate()([level_output_layers[level_number], x])
            x = self.create_localization_module(
                x, current_grid, level_filters[level_number]
            )

        x = Conv2D(8 * n_labels, (1, 1))(x)
        x = PeriodicPadding2D(current_grid)(x)
        x = Conv2D(4 * n_labels, (3, 3), strides=(2, 2))(x)
        x = PeriodicPadding2D(current_grid)(x)
        x = Conv2D(2 * n_labels, (3, 3), strides=(2, 2))(x)
        x = PeriodicPadding2D(current_grid)(x)
        x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(32)(x)
        return Dense(1)(x)
