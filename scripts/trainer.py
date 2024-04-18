import json
import logging
import os
import time
import math

import numpy as np

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Conv2D,
    Conv3D,
    Concatenate,
    DepthwiseConv1D,
    DepthwiseConv2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    LayerNormalization,
    MultiHeadAttention,
    Multiply,
    AveragePooling3D,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
)
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import get_custom_objects

from . import Core
from .utils import setup_logging, log_source
from .utils.plots import plot_histogram, plot_predictions
from .utils.tf.dataloaders import *
from .utils.tf.plots import plot_metrics
from .utils.tf.callbacks import TimedLoggingCallback, WarmupLearningRate


def alm_model(
    inputs,
    dropout_rate=0.3,
    name="",
    mha_initializer=TruncatedNormal(stddev=0.02),
    depth=3,
):
    layer = inputs
    lmax = inputs.shape[-1]
    kr = tf.keras.regularizers.l2(1e-6)

    for d in range(depth):
        x = MultiHeadAttention(
            num_heads=4,
            key_dim=lmax,
            kernel_initializer=mha_initializer,
            dropout=dropout_rate
        )(
            layer,
            layer,
            use_causal_mask = d == 0
        )
        # Note: use the Add layer to ensure that Keras masks are propagated (the + operator does not).
        layer = Add()([layer, x])
        layer = LayerNormalization()(layer)
        
        # FF layer
        x = Dense(1024, activation='relu')(layer)
        x = Dense(d_model)(x)
        x = Dropout(dropout_rate)(x)
        layer = Add()([layer, x])
        # layer = LayerNormalization()(x)


    # Now we do a final FF to get the output as a scalar
    layer = Flatten()(layer)
    out_layer = condense_layer(layer)

    return Model(inputs=input_layer, outputs=out_layer, name=name)


def alm_modelV3(input_layer, name="", dropout_rate=0.1, depth=1, heads=32):
    layer = input_layer

    # shape = tf.shape(input_layer)
    # new_shape = tf.concat([shape[:1], [-1], shape[3:]], axis=0)
    # layer = tf.reshape(input_layer, new_shape)

    d_model = layer.shape[-1]

    max_level = math.floor(math.log(d_model, 2))
    depth = min(depth, max_level)
    for level in range(depth):
        # layer = EncoderLayer(heads, 1024, 2048, dropout_rate)(layer)
        x = MultiHeadAttention(
            num_heads=heads,
            key_dim=d_model // heads,
            dropout=dropout_rate,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            # attention_axes=(1, 2),
        )(layer, layer)
        layer = Add()([layer, x])
        layer = LayerNormalization()(layer)

        # FF
        x = Dense(2 ** (max_level + 2 - level), activation="relu")(layer)
        x = Dense(d_model)(x)
        x = Dropout(dropout_rate)(x)
        layer = Add()([layer, x])
        layer = LayerNormalization()(layer)

        # redux
        d_model = 2 ** (max_level - 2 * level)
        layer = Dense(d_model)(layer)

    layer = Flatten()(layer)
    layer = Dropout(dropout_rate)(layer)
    # layer = Dense(512, activation="relu")(layer)
    layer = Dense(1, activation="relu")(layer)

    return Model(inputs=input_layer, outputs=layer, name=name)

def alm_modelV4(input_layer, name="", droupout_rate=0.1):
    layer = input_layer

    def conv(layer, filters, kernel, depth=1, pooling_size=2):
        for _ in range(depth):
            layer = Conv2D(filters, kernel, padding='same')(layer)
            layer = Conv2D(filters, kernel, padding='same')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(droupout_rate)(layer)

        layer = MaxPooling2D(pooling_size)(layer)
        return layer

    layer = Dense(32, activation='relu')(layer)
    layer = conv(layer, 64, 7, 1, 4)
    # layer = conv(layer, 64, 7, 1, 4)
    layer = conv(layer, 128, 5, 1, 4)
    # layer = conv(layer, 256, 5, 1)
    layer = conv(layer, 256, 3, 1)
    # layer = conv(layer, 512, 3, 2)

    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(256)(layer)
    layer = Dense(1)(layer)
    return Model(inputs=input_layer, outputs=layer, name=name)

def patch_modelV1(input_layer, name="", droupout_rate=0.1):
    # taken from nagarajappa
    layer = input_layer

    layer = Conv2D(32, 3, activation='relu')(layer)
    layer = MaxPooling2D(2)(layer)

    layer = Conv2D(64, 3, activation='relu')(layer)
    layer = MaxPooling2D(2)(layer)

    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(1)(layer)

    return Model(inputs=input_layer, outputs=layer, name=name)

if __name__ == "__main__":
    logger = setup_logging("trainer")

    s = Core()

    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    logger.info(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")

    # since I am changing everything so much just print the code used to create the model
    # log_source(alm_modelV2)

    MAX_EPOCHS = 1000
    BATCH_SIZE = 1

    # just some info for the model name
    timestamp = int(time.time())

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": s.sjob,
        "start_time": timestamp,
        "max_epochs": MAX_EPOCHS,
        "comment": """attention only model with alm inputs""",
    }
    logger.info(f"Extra info:\n{json.dumps(extra_info, indent=2)}")

    # settings that get passed into the model
    model_settings = {
        # "dropout_rate": 0.1,
        "name": f"{extra_info['slurm_job_id']}_Attn_Alm_{s.base_name}-{timestamp}",
        # "mha_initializer": None,
    }
    logger.info(f"Model settings:\n{json.dumps(model_settings, indent=2)}")

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": True,
        "shuffle_buffer": 1000,
    }
    logger.info(f"Data loader settings:\n{json.dumps(data_loader_args, indent=2)}")

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error"]
    logger.info(f"Looking at additional metrics: {metrics}")

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=30,
        ),
        # model checkpointing to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
            initial_value_threshold=40000,  # mse
        ),
        # custom logger to work a little better with text logs
        TimedLoggingCallback(print_frequency=60),
        # tensorboard for visualisation
        # TensorBoard(
        #     log_dir=f"{s.tb_dir}/{model_settings['name']}",
        #     histogram_freq=1,
        # ),
        TerminateOnNaN(),
    ]

    # enable wandb, set to false if not using
    if False:
        import wandb
        from wandb.keras import WandbMetricsLogger

        wandb.tensorboard.patch(root_logdir=s.tb_dir)

        wandb.init(
            project="mlpng",
            notes=extra_info["comment"],
            tags=["attn-alm", "dev"],
            config=s.settings | model_settings | extra_info,
            dir="data",
            sync_tensorboard=True,
        )

        # Add the wandb logger to the callbacks, so it is used
        callbacks.append(WandbMetricsLogger())

    # load the data and split it
    data_loader = AlmLoaderV2(s.alm_file, **data_loader_args)
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    # create and compile the model, needs to be in scope of the strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        lr_schedule = WarmupLearningRate()
        opt = Adam(learning_rate=lr_schedule)

        model = alm_modelV2(Input(data_loader.shape), **model_settings)  # type: ignore
        model.compile(optimizer=opt, loss=tf.keras.losses.mse, metrics=metrics)
        model.summary()

    # Finally, lets fit our model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=0,  # since we are using the custom logger
    )

    # Lets plot the predictions from the unseen test set
    y_pred = model.predict(test_dataset, verbose=0).flatten()
    y_test = np.concatenate([y.numpy() for _, y in test_dataset])

    # Plot the loss curves and metrics
    plot_dir = os.path.join(s.plot_dir, "training")
    file_base = os.path.join(plot_dir, model_settings["name"])
    os.makedirs(plot_dir, exist_ok=True)

    plot_metrics(history, f"{file_base}-metrics.png", metrics=["loss"] + metrics)
    plot_predictions(y_test, y_pred, f"{file_base}-preds.png")
    plot_histogram(y_test, y_pred, f"{file_base}-hist.png")
