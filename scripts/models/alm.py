import logging
import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Flatten,
    LayerNormalization,
    MultiHeadAttention,
)

from scripts.models import ModelBase, register_model
from scripts.utils import setup_logging
from scripts.utils.tf.dataloaders import AlmLoader

logger = setup_logging(__name__, level=logging.DEBUG)


@register_model
class ALM(ModelBase):
    BATCH_SIZE = 16

    def __init__(self, core, dataset_class=AlmLoader):
        super().__init__(core, dataset_class, core.alm_file)

    def _model(
        self,
        inputs,
        dropout_rate=0.3,
        mha_initializer=TruncatedNormal(stddev=0.02),
        depth=2,
        ff_density=1024,
    ):
        layer = inputs
        _, rc, lmax, _ = inputs.shape
        layer = tf.reshape(layer, (-1, rc * lmax, lmax))

        for d in range(depth):
            x = MultiHeadAttention(
                num_heads=4,
                key_dim=lmax,
                kernel_initializer=mha_initializer,
                dropout=dropout_rate,
            )(layer, layer)
            layer = Add()([layer, x])
            layer = LayerNormalization()(layer)

            # FF layer
            x = Dense(ff_density, activation="relu")(layer)
            x = Dense(lmax)(x)
            x = Dropout(dropout_rate)(x)
            layer = Add()([layer, x])
            layer = LayerNormalization()(x)

        # Now we do a final FF to get the output as a scalar
        layer = Flatten()(layer)
        # layer = Dense(512)(layer)
        # layer = Dense(128)(layer)
        # layer = Dense(64)(layer)
        return Dense(1)(layer)
