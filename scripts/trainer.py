import os
import time
import inspect
import re
import json
import logging

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# I was getting double logging from tensorflow, this stops that
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MultiHeadAttention,
    Add,
    Multiply,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal

import Core
from utils import setup_logging, log_source
from utils.tf.dataloaders import *
from utils.tf.plots import plot_histogram, plot_metrics, plot_predictions
from utils.tf.callbacks import TimedLoggingCallback, WarmupLearningRate


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
        x = Dense(1024, 'relu')(layer)
        x = Dense(lmax)(x)
        x = Dropout(dropout_rate)(x)
        x = Add()([layer, x])
        res = layer = LayerNormalization()(x)

    # Now we do a final FF to get the output as a scalar
    layer = Flatten()(layer)
    layer = Dense(1024, 'sigmoid', kernel_regularizer=kr)(layer)
    layer = Dense(512)(layer)
    layer = Dense(256)(layer)
    layer = Dense(1)(layer)
    return Model(inputs=inputs, outputs=layer, name=name)


if __name__ == "__main__":
    logger = setup_logging("trainer")

    s = Core()

    # print out the versions of the libraries
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    logger.info(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")

    # since I am changing everything so much just print the code used to create the model
    log_source(alm_model)

    MAX_EPOCHS = 1000
    BATCH_SIZE = 8

    # just some info for the model name
    timestamp = int(time.time())

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID", 0),
        "start_time": timestamp,
        "max_epochs": MAX_EPOCHS,
        "comment": """attention only model with alm inputs""",
    }
    logger.info(f"Extra info:\n{json.dumps(extra_info, indent=2)}")

    # settings that get passed into the model
    model_settings = {
        "dropout_rate": 0.3,
        "name": f"{extra_info['slurm_job_id']}_Attn_Alm_{s.base_name}-{timestamp}",
        # "mha_initializer": None,
    }
    logger.info(f"Model settings:\n{json.dumps(model_settings, indent=2)}")

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": False,
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
        # model checkpoining to save the best model
        # ModelCheckpoint(
        #     f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
        #     monitor="val_loss",
        #     save_best_only=True,
        #     mode="auto",
        #     initial_value_threshold=40000, # mse
        # ),
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
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

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
        lr_schedule = WarmupLearningRate(
            warmup_learning_rate=1e-7,  # start small
            warmup_steps=5000,
            warmup_scale=150,
            warmup_scale_steps=1,
            warmed_learning_rate=1e-3,
            decay_steps=1e5,
            decay_rate=0.95,
            staircase=True,
        )
        opt = Adam(learning_rate=lr_schedule)

        model = alm_model(Input(data_loader.shape), **model_settings)
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
