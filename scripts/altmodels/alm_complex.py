import os
import pprint
import sys
import time
from math import floor

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from healpy.sphtfunc import Alm, alm2map

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

from utils import Config
from utils.tf import dice_coefficient_loss
from utils.tf.callbacks import TimedLoggingCallback
from utils.tf.plots import plot_histogram, plot_metrics, plot_predictions

from utils.tf.dataloaders import AlmLoaderTFDS

import cvnn.layers as complex_layers


def alm_model(
    inputs,
    optimizer=Adam,
    metrics=[],
    dropout_rate=0.3,
    loss_function=dice_coefficient_loss,
    name="",
):
    # real_input = tf.math.real(inputs)
    # imag_input = tf.math.imag(inputs)

    # # Process the real and imaginary parts separately
    # real_layer = Conv1D(32, 9, padding='same')(real_input)
    # real_layer = Conv1D(64, 9, padding='same')(real_input)
    # real_layer = Activation('relu')(real_layer)
    # real_out_layer = Dropout(dropout_rate)(real_layer)

    # imag_layer = Conv1D(32, 9, padding='same')(imag_input)
    # imag_layer = Conv1D(64, 9, padding='same')(imag_input)
    # imag_layer = Activation('relu')(imag_layer)
    # imag_out_layer = Dropout(dropout_rate)(imag_layer)

    layer = complex_layers.ComplexConv1D(
        16, 9, padding="valid", activation="cart_relu"
    )(inputs)
    layer = complex_layers.ComplexConv1D(
        32, 9, padding="valid", activation="cart_relu"
    )(layer)
    layer = complex_layers.ComplexConv1D(
        64, 9, padding="valid", activation="cart_relu"
    )(layer)
    layer = complex_layers.ComplexAvgPooling1D(2)(layer)

    layer = complex_layers.ComplexConv1D(
        128, 9, padding="valid", activation="cart_relu"
    )(layer)
    layer = complex_layers.ComplexConv1D(
        64, 9, padding="valid", activation="cart_relu"
    )(layer)
    layer = complex_layers.ComplexConv1D(
        32, 9, padding="valid", activation="cart_relu"
    )(layer)
    layer = complex_layers.ComplexAvgPooling1D(2)(layer)

    # Concatenate the real and imaginary parts
    # out_layer = Concatenate()([real_out_layer, imag_out_layer])
    # layer =  complex_layers.ComplexConv1D(1, 9, padding='same', activation='cart_relu')(layer)
    layer = complex_layers.ComplexFlatten()(layer)
    layer = complex_layers.ComplexDense(512)(layer)
    layer = complex_layers.ComplexDense(256)(layer)
    layer = complex_layers.ComplexDense(1, activation="convert_to_real_with_abs")(layer)

    model = Model(inputs=inputs, outputs=layer, name=name)

    # finally we compile the model
    # needs to be done in strategy scope
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model


if __name__ == "__main__":
    s = Config(sys.argv[1:])

    MAX_EPOCHS = 300
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
        "comment": """alm test""",
    }

    model_settings = {
        "dropout_rate": 0.3,
        "name": f"{extra_info['slurm_job_id']}_Alm_{s.base_name}-{timestamp}",
    }

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": True,
    }

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error", "mse"]

    # enable a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=model_settings["initial_learning_rate"],
        decay_steps=10,
        decay_rate=0.95,
        staircase=True,
    )

    # callbacks to use during training
    callbacks = [
        # # We use earlystoping to prevent overfitting
        # EarlyStopping(
        #     monitor="val_loss",
        #     patience=10,
        #     verbose=1,
        #     restore_best_weights=True,
        #     start_from_epoch=0,
        # ),
        # model checkpoining to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
        # TimedLoggingCallback(), # custom logger to work a little better with text logs
    ]

    # enable wandb, set to false if not using
    if False:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

        wandb.init(
            project="mlpng",
            notes=extra_info["comment"],
            tags=["alm", "dev"],
            config=s.settings | model_settings | extra_info,
            dir="data",
            sync_tensorboard=True,
        )

        # Add the wandb logger to the callbacks, so it is used
        callbacks.append(WandbMetricsLogger())

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input = Input((Alm.getsize(s.lmax), 1), name="input", dtype=tf.complex64)
        # input = complex_layers.complex_input((Alm.getsize(s.lmax), 1), name="input")

        opt = Adam(
            learning_rate=lr_schedule,
            # learning_rate=model_settings["initial_learning_rate"],
        )

        model = alm_model(input, opt, metrics, **model_settings)

    # Lets load our data
    tfds_filepath = s.alm_file.replace(".hdf5", ".tfds")
    data_loader = AlmLoaderTFDS(tfds_filepath, **data_loader_args)
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    pprint.PrettyPrinter(indent=2).pprint(model_settings | extra_info)
    model.summary()

    # Finally, lets fit our model
    # we use the train and val sets here, so the model will not see the test set
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,  # since we are using the custom logger
    )

    # Lets plot the predictions from the unseen test set
    y_pred = model.predict(test_dataset)
    y_test = np.concatenate([y.numpy() for _, y in test_dataset])

    # Plot the loss curves and metrics
    plot_metrics(
        history,
        f"{s.plot_dir}/{model_settings['name']}-metrics.png",
        metrics=["loss"] + metrics,
    )
    plot_predictions(y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-preds.png")
    plot_histogram(
        y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-histogram.png"
    )
