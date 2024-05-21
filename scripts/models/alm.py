import logging
import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Flatten,
    GroupNormalization,
    LayerNormalization,
    BatchNormalization,
    MultiHeadAttention,
)

from scripts.models import ModelCore, register_model
from scripts.utils import setup_logging
from scripts.utils.tf import AlmLoader

logger = setup_logging(__name__, level=logging.DEBUG)


@register_model
class ALM(ModelCore):
    """This model attempts to use a transformer to predict the fnl value from the alm data."""

    def __init__(self, argv=None):
        super().__init__(argv)

        if self.lmax >= 1024:
            self.BATCH_SIZE = 16
        else:
            self.BATCH_SIZE = 32

    def init_dataset(self, *args, **kwargs):
        self._dataset = AlmLoader(self.alm_file, *args, **kwargs)
        return self._dataset

    def _model(
        self,
        inputs,
        dropout_rate=0.3,
        depth=3,
        ff_density=512,
        mha_num_heads=8,
        mha_dropout=0.3,
    ):
        layer = inputs
        layer = tf.reshape(layer, (-1, self.lmax, 2 * self.lmax))

        # Some fully connected layers and down sampling to get the model to a reasonable size
        # the last layer is the m's with real and complex values, but will have 0 for half the values
        # This should give us a good amount of room to reduce the size without much impact on the model
        layer = Dense(2 * self.lmax)(layer)
        layer = Dense(self.lmax)(layer)
        layer = Dense(self.lmax // 8)(layer)
        d_model = layer.shape[-1]  # // mha_num_heads

        for _ in range(depth):
            x = MultiHeadAttention(
                num_heads=mha_num_heads,
                key_dim=d_model,
                dropout=mha_dropout,
                # kernel_initializer=TruncatedNormal(stddev=0.01),
                # bias_initializer=TruncatedNormal(stddev=0.01),
            )(layer, layer)
            layer = Add()([layer, x])
            # this normalization applies to all the data vs just a single channel
            layer = GroupNormalization(groups=-1)(layer)

            # FF layer
            x = Dense(ff_density, activation="relu")(layer)
            x = Dense(d_model)(x)
            x = Dropout(dropout_rate)(x)
            layer = Add()([layer, x])
            layer = GroupNormalization(groups=-1)(layer)

        # Now we do a final FF to get the output as a scalar
        layer = Flatten()(layer)
        # layer = Dense(512)(layer)
        # layer = Dense(128)(layer)
        layer = Dense(64, activation="relu")(layer)
        return Dense(1)(layer)
