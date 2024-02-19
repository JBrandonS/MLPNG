import os
import pprint
import sys
import time

import numpy as np
from healpy.sphtfunc import Alm

# os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.environ['CUDA_HOME']}"

import tensorflow as tf
from dataloaders import AlmLoader
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (Activation, AveragePooling1D,
                                     BatchNormalization, Concatenate, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D, Lambda,
                                     LayerNormalization, MaxPooling2D,
                                     MultiHeadAttention, Multiply)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from utils import SimConfig
from utils.tf import TimedLoggingCallback, dice_coefficient_loss
from utils.tf.layers import PeriodicPadding2D
from utils.tf.plots import plot_histogram, plot_metrics, plot_predictions


def alm_model(
    inputs,
    dropout_rate=0.3,
    name="",
):
    input_layer = inputs

    layer = MultiHeadAttention(
        num_heads=1, key_dim=32, dropout=dropout_rate, attention_axes=(3)
    )(input_layer, input_layer)
    layer = MultiHeadAttention(
        num_heads=1, key_dim=32, dropout=dropout_rate, attention_axes=(2)
    )(input_layer, layer)
    layer = MultiHeadAttention(
        num_heads=1, key_dim=32, dropout=dropout_rate, attention_axes=(1)
    )(input_layer, layer)

    # layer = MultiHeadAttention(num_heads=1, key_dim=8, dropout=dropout_rate, attention_axes=(1, 3))(input_layer, layer)
    # layer = MultiHeadAttention(num_heads=1, key_dim=8, dropout=dropout_rate, attention_axes=(2, 3))(input_layer, layer)

    layer = Conv2D(1, (1, 1))(layer)
    layer = PeriodicPadding2D(layer.shape[1])(layer)
    layer = Conv2D(1, (3, 3), strides=(2, 2))(layer)
    layer = PeriodicPadding2D(layer.shape[1])(layer)
    layer = Conv2D(1, (3, 3), strides=(2, 2))(layer)
    layer = PeriodicPadding2D(layer.shape[1])(layer)
    layer = Conv2D(1, (3, 3), strides=(2, 2))(layer)

    layer = Flatten()(layer)
    layer = Dense(1)(layer)

    return Model(inputs=inputs, outputs=layer, name=name)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print("TensorFlow version:", tf.__version__)
    print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    print(f"Number of GPUs Available: {len(gpus)}")

    s = SimConfig(sys.argv[1])

    MAX_EPOCHS = 100
    BATCH_SIZE = 1

    # just some info for the model name
    timestamp = int(time.time())
    lens_str = "lensed" if s.lensing else "unlensed"

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID") or 0,
        "start_time": timestamp,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "comment": """attention only alm test""",
    }

    model_settings = {
        # "initial_learning_rate": 1e-3,
        "name": f"{extra_info['slurm_job_id']}_Attn-Alm_{s.base_name}-{timestamp}",
    }

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": True,
    }

    print("Model Settings:")
    pprint.PrettyPrinter(indent=2).pprint(
        model_settings | extra_info | data_loader_args
    )

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error"]

    # enable a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,  # model_settings["initial_learning_rate"],
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
            start_from_epoch=0,
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
        TensorBoard(log_dir=f"{s.tb_dir}/{model_settings['name']}", histogram_freq=1),
    ]

    # enable wandb, set to false if not using
    if True:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

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

    data_loader = AlmLoader(s.alm_file_complete, **data_loader_args)
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input = Input(data_loader.shape)

        model = alm_model(input, name=model_settings["name"])

        opt = Adam(
            learning_rate=lr_schedule,
            # learning_rate=model_settings["initial_learning_rate"],
        )

        # finally we compile the model
        # needs to be done in strategy scope
        model.compile(
            optimizer=opt,
            loss="mse",
            metrics=metrics,
        )

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
