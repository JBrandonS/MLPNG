import os
import pprint
import sys
import time
import inspect
import re
import json

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MultiHeadAttention,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

from utils import Config, setup_logging
from utils.tf.dataloaders import AlmLoader
from utils.tf.plots import plot_histogram, plot_metrics, plot_predictions
from utils.tf.callbacks import TimedLoggingCallback, WarmupLearningRate


def alm_model(
    inputs,
    dropout_rate=0.3,
    name="",
    mha_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
):
    input_layer = inputs

    # We start with data in (batch,2,500,500), (real/complex, l, m)
    # This is too much data for a transformer so we collapse down the m axis using a FF
    # We could look at something smarter to do this
    # then we squeeze to get a final data size of (2, 500) which attention can handle
    layer = Dense(512, activation="sigmoid")(input_layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(128, activation="sigmoid")(layer)
    layer = Dense(1)(layer)
    layer = tf.squeeze(layer, axis=-1)  # (batch, 2, 500)

    layer = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        kernel_initializer=mha_initializer,
        dropout=dropout_rate,
    )(layer, layer)

    # Now we do a final FF to get the output as a scalar
    layer = Flatten()(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(1024, activation="relu", kernel_initializer="he_uniform")(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(128)(layer)
    layer = Dense(1)(layer)

    return Model(inputs=inputs, outputs=layer, name=name)


if __name__ == "__main__":
    logger = setup_logging("trainer")

    # print out the versions of the libraries
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    logger.info(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")

    # since I am changing everything so much just print the code used to create the model
    source = inspect.getsource(alm_model)
    source = re.sub(r"#.*", "", source)
    source = re.sub(r"\n\s*\n", "\n", source)
    logger.info(f"Model source:\n{source}")

    # load the config
    s = Config(sys.argv[1:])

    MAX_EPOCHS = 300
    BATCH_SIZE = 32

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
        "cache": True,
        "shuffle_buffer": 1000,
    }
    logger.info(f"Data loader settings:\n{json.dumps(data_loader_args, indent=2)}")

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error"]
    logger.info(f"Looking for additional metrics: {metrics}")

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        # EarlyStopping(
        #     monitor="val_loss",
        #     patience=20,
        #     verbose=1,
        #     restore_best_weights=True,
        #     start_from_epoch=50
        # ),
        # model checkpoining to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
            initial_value_threshold=40000,
        ),
        # custom logger to work a little better with text logs
        TimedLoggingCallback(print_frequency=60),
        # tensorboard for visualisation
        TensorBoard(
            log_dir=f"{s.tb_dir}/{model_settings['name']}",
            histogram_freq=1,
        ),
    ]

    # enable wandb, set to false if not using
    if True:
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

    # get our strategy to allow multi-gpu training
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    logger.info(f"Number of GPUs Available: {num_gpus}")

    # load the data and split it
    data_loader = AlmLoader(s.alm_file, num_replicas=num_gpus, **data_loader_args)
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    # create and compile the model, needs to be in scope of the strategy
    with strategy.scope():
        lr_schedule = WarmupLearningRate(
            warmup_learning_rate=1e-8,  # start small
            warmup_steps=1e6,
            warmup_scale=1.5,
            warmup_scale_steps=1,
            warmed_learning_rate=1e-3,
            decay_steps=100000,
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
