import os
import sys
import time
import pprint

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})

from keras import Input, Model
from keras.layers import (
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
from keras.optimizers import Adam

from tensorflow import pad

from utils import Config
from utils.tf import TimedLoggingCallback
from scripts.utils.tf.dataloaders import PatchLoader
from utils.plots import plot_histogram, plot_metrics, plot_predictions

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ExportCallback,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# from tensorflow_addons.layers import InstanceNormalization


class PeriodicPadding2D(Layer):
    def __init__(self, current_grid, **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.current_grid = current_grid
        self.indices = np.append(
            np.insert(np.arange(self.current_grid), 0, self.current_grid - 1), 0
        ).astype(np.int64)

    def call(self, x):
        x = tf.gather(x, self.indices, axis=1)
        x = tf.gather(x, self.indices, axis=2)
        return x


def create_localization_module(input_layer, current_grid, n_filters):
    layer1 = PeriodicPadding2D(current_grid)(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1))
    return convolution2


def create_up_sampling_module(input_layer, current_grid, n_filters, size=(2, 2)):
    up_sample = UpSampling2D(size=size)(input_layer)
    layer1 = PeriodicPadding2D(current_grid)(up_sample)
    convolution = create_convolution_block(layer1, n_filters)
    return convolution


def create_context_module(
    input_layer,
    current_grid,
    n_level_filters,
    dropout_rate=0.3,
    data_format="channels_last",
):
    layer1 = PeriodicPadding2D(current_grid)(input_layer)
    convolution1 = create_convolution_block(
        input_layer=layer1, n_filters=n_level_filters
    )
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = PeriodicPadding2D(current_grid)(dropout)
    convolution2 = create_convolution_block(
        input_layer=layer2, n_filters=n_level_filters
    )
    return convolution2


def create_convolution_block(
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


def UNET(
    image_size,
    n_base_filters=16,
    depth=5,
    dropout_rate=0.3,
    n_labels=1,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=tf.keras.losses.mse,
):

    inputs = Input((image_size, image_size, 1), name="img")
    x = inputs

    level_output_layers = list()
    level_filters = list()

    current_grid = image_size

    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if x is inputs:
            x = PeriodicPadding2D(current_grid)(x)
            x = create_convolution_block(x, n_level_filters)
        else:
            x = PeriodicPadding2D(current_grid)(x)
            x = create_convolution_block(x, n_level_filters, strides=(2, 2))
            current_grid /= 2

        previous_block = x
        x = create_context_module(
            x, current_grid, n_level_filters, dropout_rate=dropout_rate
        )
        x = Add()([previous_block, x])

        level_output_layers.append(x)

    for level_number in range(depth - 2, -1, -1):
        current_grid *= 2
        x = create_up_sampling_module(x, current_grid, level_filters[level_number])
        x = Concatenate()([level_output_layers[level_number], x])
        x = create_localization_module(x, current_grid, level_filters[level_number])

    x = Conv2D(n_labels, (1, 1))(x)
    x = PeriodicPadding2D(current_grid)(x)
    x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)
    x = PeriodicPadding2D(current_grid)(x)
    x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)
    x = PeriodicPadding2D(current_grid)(x)
    x = Conv2D(n_labels, (3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer(learning_rate=initial_learning_rate),
        loss=loss_function,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )
    return model


if __name__ == "__main__":
    s = Config(sys.argv[1:])

    MAX_EPOCHS = 100
    BATCH_SIZE = 32

    # just some info for the model name
    timestamp = int(time.time())
    lens_str = "lensed" if s.lensing else "unlensed"

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID") or 0,
        "start_time": timestamp,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "comment": """Isensee model taken from NN-pnG 2/17/24""",
    }

    model_settings = {
        "depth": 5,
        "dropout_rate": 0.3,
        "loss_function": tf.keras.losses.mse,
        "initial_learning_rate": 1e-3,
        "name": f"{extra_info['slurm_job_id']}_isensee_{s.base_name}-{timestamp}",
    }

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": True,
    }

    # enable a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=model_settings["initial_learning_rate"],
        decay_steps=3,
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
            start_from_epoch=0,
        ),
        # Model checkpoining to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
        TimedLoggingCallback(
            print_frequency=60
        ),  # custom logger to work a little better with text logs
        TensorBoard(log_dir=f"{s.tb_dir}/{model_settings['name']}", histogram_freq=1),
    ]

    # enable wandb, set to false if not using
    if False:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

        wandb.init(
            project="mlpng",
            notes=extra_info["comment"],
            tags=["isensee", "dev"],
            config=s.settings | model_settings | extra_info,
            dir="data",
            sync_tensorboard=True,
        )

        # Add the wandb logger to the callbacks, so it is used
        callbacks.append(WandbMetricsLogger())

    # Lets load our data
    data_loader = PatchLoader(s.patch_file, **data_loader_args)
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        opt = Adam(learning_rate=lr_schedule)

        model = UNET(data_loader.shape[1], optimizer=opt, **model_settings)

    pprint.PrettyPrinter(indent=2).pprint(model_settings | extra_info)
    model.summary()

    # Finally, lets fit our model
    # we use the train and val sets here, so the model will not see the test set
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=0,  # since we are using the custom logger
    )

    # Lets plot the predictions from the unseen test set
    print(test_dataset.element_spec)
    y_pred = model.predict(test_dataset)
    y_test = np.concatenate([y.numpy() for _, y in test_dataset])
    print(y_pred.shape, y_test.shape)

    # Plot the loss curves and metrics
    plot_metrics(
        history,
        f"{s.plot_dir}/{model_settings['name']}-metrics.png",
        metrics=["loss"],
    )
    plot_predictions(y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-preds.png")
    plot_histogram(
        y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-histogram.png"
    )
