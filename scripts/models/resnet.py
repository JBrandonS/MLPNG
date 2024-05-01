import math

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Flatten,
)

from scripts.models import ModelBase, register_model
from scripts.utils.tf.layers import augmentation_layer


@register_model
class RESNET(ModelBase):
    def __init__(self, core):
        super().__init__(core)

        # this finds the number of factor of 2 reductions in the spatial dimensions to make final depth 16x16
        # just to ensure we don't go too deep / small
        self.max_depth = math.log(core.nside / 32, 2) + 1

    def _model(
        self,
        inputs,
        dropout_rate=0.3,
        flip=False,
        rotate=False,
        add_powers=2,
    ):

        input_shape = (self.core.nside, self.core.nside, 3)  # has to be 3 channels
        resnet50 = ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        resnet50.trainable = False
        
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

        # allows the model to operate on a -1,1 scale
        return out_layer * 1000
