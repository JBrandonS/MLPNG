import os
import pprint
import sys
import time
import inspect
import re

import numpy as np
from healpy.sphtfunc import Alm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    Flatten,
    MultiHeadAttention,
    Concatenate,
    LayerNormalization,
    Dropout,
    Add,
    Reshape,
    AveragePooling2D,
    AveragePooling3D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

from utils import Config
from utils.tf.dataloaders import AlmLoader
from utils.tf.layers import PeriodicPadding2D
from utils.tf.plots import plot_histogram, plot_metrics, plot_predictions
from utils.tf.callbacks import TimedLoggingCallback, WarmupLearningRate


def simple_transformer(inputs, dropout_rate=0.3, name=""):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

    x = MultiHeadAttention(
        num_heads=1,
        key_dim=1,
        kernel_initializer=initializer,
    )(inputs, inputs)


    layer = Add()([inputs, x])
    layer = LayerNormalization()(layer)
    layer = Flatten()(layer)
    layer = Dense(1)(layer)

    return Model(inputs=inputs, outputs=layer, name=name)

def alm_model(
    inputs,
    dropout_rate=0.3,
    name="",
    mha_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
):
    input_layer = inputs

    # remove the last axis
    layer = Dense(512, activation='sigmoid')(input_layer)
    layer = Dense(128, activation='sigmoid')(layer)
    layer = Dense(1)(layer)
    layer = tf.squeeze(layer, axis=-1) # (, 2, 500)

    layer = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        kernel_initializer=mha_initializer,
        dropout=dropout_rate,
    )(layer, layer)

    # layer = MultiHeadAttention(
    #     num_heads=8,
    #     key_dim=n_features,
    #     kernel_initializer=initializer,
    #     dropout=dropout_rate,
    # )(layer, layer)

    layer = Flatten()(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(1024, activation="relu")(layer)
    layer = Dense(128)(layer)
    layer = Dense(1)(layer)

    return Model(inputs=inputs, outputs=layer, name=name)


if __name__ == "__main__":
    s = Config(sys.argv[1])

    MAX_EPOCHS = 300
    BATCH_SIZE = 1

    # just some info for the model name
    timestamp = int(time.time())

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID") or 0,
        "start_time": timestamp,
        "max_epochs": MAX_EPOCHS,
        "comment": """attention only model with alm inputs""",
    }

    model_settings = {
        "dropout_rate": 0.3,
        "name": f"{extra_info['slurm_job_id']}_Attn_Alm_{s.base_name}-{timestamp}",
        # "mha_initializer": None,
    }

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": True,
        "shuffle_buffer": 1000,
        "normalize": False,
        "dtype": np.float32,
    }

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error"]

    lr_schedule = WarmupLearningRate(
        warmup_learning_rate=1e-8,  # start small
        warmup_steps=1e5,
        warmup_scale=5,
        warmup_scale_steps=1,
        warmed_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True,
    )

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=50,
        ),
        # model checkpoining to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
        # custom logger to work a little better with text logs
        TimedLoggingCallback(print_frequency=60),
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

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        num_gpus = strategy.num_replicas_in_sync

        data_loader = AlmLoader(
            s.alm_file, flat_output=False, num_replicas=num_gpus, **data_loader_args
        )
        train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

        opt = Adam(learning_rate=lr_schedule)

        # prints the source of the model, mostly for debugging
        source = inspect.getsource(alm_model)
        source = re.sub(r"#.*", "", source)
        source = re.sub(r"\n\s*\n", "\n", source)
        tf.print("model source code: \n" + source)

        model = alm_model(Input(data_loader.shape), **model_settings)
        model.compile(optimizer=opt, loss=tf.keras.losses.mse, metrics=metrics)

    # print some logging into before we start training
    source = inspect.getsource(alm_model)
    source = re.sub(r"#.*", "", source)
    source = re.sub(r"\n\s*\n", "\n", source)
    tf.print("Model source: \n" + source)

    pp = pprint.PrettyPrinter(indent=2)
    tf.print(f"TensorFlow version: {tf.__version__}")
    tf.print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    tf.print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    tf.print(f"Number of GPUs Available: {num_gpus}")

    pp.pprint(model_settings)
    pp.pprint(data_loader_args)
    pp.pprint(extra_info | {"optimizer": opt, "metrics": metrics})
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
    plot_metrics(
        history,
        f"{s.plot_dir}/{model_settings['name']}-metrics.png",
        metrics=["loss"] + metrics,
    )
    plot_predictions(y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-preds.png")
    plot_histogram(
        y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-histogram.png"
    )
