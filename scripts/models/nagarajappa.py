from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from scripts.models import ModelBase, register_model


@register_model
class NAGARAJAPPA(ModelBase):
    """A CNN model based on Nagarajappa and Ma, 2024"""

    def _model(self, inputs):
        layer = Conv2D(32, 3, activation="relu")(inputs)
        layer = MaxPooling2D(2)(layer)

        layer = Conv2D(64, 3, activation="relu")(layer)
        layer = MaxPooling2D(2)(layer)

        layer = Flatten()(layer)
        layer = Dense(256, activation="relu")(layer)
        layer = Dense(128, activation="relu")(layer)
        layer = Dense(64, activation="relu")(layer)
        return Dense(1)(layer)
