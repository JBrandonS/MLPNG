import os
import sys
import time

import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.losses import mse
from tensorflow.keras.layers import (
    Layer,
    LeakyReLU,
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
from tensorflow.keras.optimizers import Adam

from scripts.models import AutoModel, ModelCore, register_model
from scripts.utils import plot_histogram, plot_predictions
from scripts.utils.tf import plot_metrics, try_init_wandb

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay


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
class ISENSEE(ModelCore):
    """This is the model taken from Thomas' UNET_fnl notebook"""

    BATCH_SIZE = 16

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
        up_sample = UpSampling2D(size=size)(input_layer)
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
        activation=LeakyReLU,
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

    def _model(self, inputs, depth=3, n_base_filters=32, dropout_rate=0.3, n_labels=16):
        x = inputs
        level_output_layers = list()
        level_filters = list()
        current_grid = x.shape[1]

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

        x = Conv2D(n_labels, (1, 1))(x)
        x = PeriodicPadding2D(current_grid)(x)
        x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)
        x = PeriodicPadding2D(current_grid)(x)
        x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)
        x = PeriodicPadding2D(current_grid)(x)
        x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)

        x = Flatten()(x)
        return Dense(1)(x)


def main():
    model = AutoModel()

    MAX_EPOCHS = 100
    BATCH_SIZE = 16

    # just some info for the model name
    timestamp = int(time.time())
    slurm_job_id = os.getenv("SLURM_JOB_ID") or 0
    name = f"{slurm_job_id}_isensee_{model.base_name}_{timestamp}"

    # enable a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.95,
        staircase=True,
    )

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        # Model checkpoining to save the best model
        ModelCheckpoint(
            f"{model.model_dir}/{name}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
    ]

    # Lets load our data
    dataset = model.init_dataset(batch_size=BATCH_SIZE)
    train_ds, test_ds, val_ds = dataset.get_split(0.8, 0.1, 0.1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        opt = Adam(learning_rate=lr_schedule)
        model.make_model()
        model.compile(optimizer=opt, loss=mse)
        model.summary()

    # Finally, lets fit our model
    # we use the train and val sets here, so the model will not see the test set
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Lets plot the predictions from the unseen test set
    preds = model.predict(test_ds, verbose=1).flatten()
    truth = np.concatenate([y.numpy() for _, y in test_ds])

    # Plot the loss curves and metrics
    plot_metrics(
        history,
        f"{model.plot_dir}/{name}-metrics.png",
        metrics=["loss"],
    )
    plot_predictions(truth, preds, f"{model.plot_dir}/{name}-preds.png")
    plot_histogram(truth, preds, f"{model.plot_dir}/{name}-histogram.png")


if __name__ == "__main__":
    sys.exit(main())
