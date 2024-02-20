import os
import pprint
import sys
import time

import numpy as np
from healpy.sphtfunc import Alm

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

from utils import Config
from utils.data import AlmLoader
from utils.tf.layers import PeriodicPadding2D
from utils.tf.plots import plot_histogram, plot_metrics, plot_predictions
from utils.tf.callbacks import TimedLoggingCallback, WarmupLearningRate


def alm_model(
    inputs,
    dropout_rate=0.3,
    name="",
):
    # lets try this init method recommended online
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    # setup the multihead attention layer shared options just to keep things clear
    mha_args = {
        "num_heads": 1,
        "key_dim": 1,
        "dropout": dropout_rate,
        "kernel_initializer": initializer,
    }

    input_layer = inputs
    input_layer_transposed = tf.transpose(input_layer, perm=[0, 2, 1, 3])

    layer = MultiHeadAttention(num_heads=32, key_dim=2, dropout=dropout_rate, kernel_initializer=initializer, attention_axes=(3))(
        input_layer, input_layer_transposed, input_layer
    )
    layer = MultiHeadAttention(num_heads=4, key_dim=501, dropout=dropout_rate, kernel_initializer=initializer, attention_axes=(2))(
        input_layer, input_layer_transposed, layer
    )
    layer = MultiHeadAttention(num_heads=4, key_dim=501, dropout=dropout_rate, kernel_initializer=initializer, attention_axes=(1))(
        input_layer, input_layer_transposed, layer
    )

    layer = MultiHeadAttention(num_heads=1, key_dim=1, dropout=dropout_rate, kernel_initializer=initializer, attention_axes=(1, 3))(input_layer, layer)
    layer = MultiHeadAttention(num_heads=1, key_dim=1, dropout=dropout_rate, kernel_initializer=initializer, attention_axes=(2, 3))(input_layer, layer)
    # layer = MultiHeadAttention(**mha_args)(layer, layer)

    layer = Conv2D(256, (1, 1))(layer)
    # layer = PeriodicPadding2D(layer.shape[1])(layer)
    layer = Conv2D(126, (3, 3), strides=(2, 2))(layer)
    # layer = PeriodicPadding2D(layer.shape[1])(layer)
    layer = Conv2D(32, (3, 3), strides=(2, 2))(layer)
    # layer = PeriodicPadding2D(layer.shape[1])(layer)
    layer = Conv2D(16, (3, 3), strides=(2, 2))(layer)

    layer = Flatten()(layer)
    layer = Dense(1)(layer)

    return Model(inputs=inputs, outputs=layer, name=name)


if __name__ == "__main__":
    s = Config(sys.argv[1])

    MAX_EPOCHS = 1000
    BATCH_SIZE = 1

    # just some info for the model name
    timestamp = int(time.time())

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID") or 0,
        "start_time": timestamp,
        "max_epochs": MAX_EPOCHS,
        "comment": """attention Alm test""",
    }

    model_settings = {
        "dropout_rate": 0.3,
        "name": f"{extra_info['slurm_job_id']}_Attn-Alm_{s.base_name}-{timestamp}",
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
        warmup_learning_rate=1e-6, # start small
        warmup_steps=3000,
        warmup_scale=1.5,
        warmup_scale_steps=1,

        base_learning_rate=1e-3,
        decay_steps=1000,
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
            start_from_epoch=30,
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

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        num_gpus = strategy.num_replicas_in_sync

        data_loader = AlmLoader(
            s.alm_file_complete, num_replicas=num_gpus, **data_loader_args
        )
        train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

        opt = Adam(learning_rate=lr_schedule)
        model = alm_model(Input(data_loader.shape), name=model_settings["name"])
        model.compile(optimizer=opt, loss=tf.keras.losses.mse, metrics=metrics)

    # print some logging into before we start training
    gpus = tf.config.experimental.list_physical_devices("GPU")
    pp = pprint.PrettyPrinter(indent=2)
    tf.print(f"TensorFlow version: {tf.__version__}")
    tf.print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    tf.print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    tf.print(f"Number of GPUs Available: {len(gpus)}")
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
