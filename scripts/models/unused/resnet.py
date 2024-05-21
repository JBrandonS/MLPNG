import math

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Flatten,
)

from scripts.models import ModelCore, register_model
from scripts.utils.tf.layers import augmentation_layer


@register_model
class RESNET(ModelCore):
    def __init__(self, argv=None):
        super().__init__(argv)

        # this finds the number of factor of 2 reductions in the spatial dimensions to make final depth 16x16
        # just to ensure we don't go too deep / small
        self.max_depth = math.log(self.nside / 32, 2) + 1

    def _model(
        self,
        inputs,
        dropout_rate=0.3,
        flip=False,
        rotate=False,
        add_powers=2,
    ):

        input_shape = (self.nside, self.nside, 3)  # has to be 3 channels
        resnet50 = ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

        # After initial training
        resnet50.trainable = False

        # We chose to train the top 2 resnet blocks, i.e. we will freeze
        # the first 143 layers and unfreeze the rest:
        for layer in resnet50.layers[-2:]:
            layer.trainable = True

        layer = augmentation_layer(flip, rotate, add_powers)(inputs)
        layer = Concatenate()([layer, layer, layer])
        layer = resnet50(layer)

        # FF network
        # layer = Conv2D(1, (1, 1))(layer)
        out_layer = Flatten()(layer)
        out_layer = Dropout(dropout_rate)(out_layer)

        n_neurons = min(1024, out_layer.shape[-1])
        while n_neurons > 1:
            n_neurons = max(1, n_neurons // 32)

            if n_neurons < 64:
                # create last layer with 1 neuron, and no activation then stop
                out_layer = Dense(1, activation="sigmoid")(out_layer)
                break
            else:
                out_layer = Dense(n_neurons, activation="relu")(out_layer)

        return Dense(1)(out_layer)
