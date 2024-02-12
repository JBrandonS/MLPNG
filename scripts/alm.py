import os
import pprint
import sys
import time

import numpy as np
from healpy.sphtfunc import Alm

# os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.environ['CUDA_HOME']}"

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
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
    AveragePooling1D,
    GlobalAveragePooling1D,
    BatchNormalization,
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

from utils import SimConfig
from utils.tf import (
    TimedLoggingCallback,
    dice_coefficient_loss,
)

from utils.tf.plots import (
    plot_histogram,
    plot_metrics,
    plot_predictions,
)
from dataloaders import AlmDataLoader


def alm_model(
    inputs,
    optimizer=Adam,
    metrics=[],
    dropout_rate=0.3,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="",
):
    real_input = inputs[:, :, :, 0]
    imag_input = inputs[:, :, :, 1]

    # Process the real and imaginary parts separately
    cnn_block = Sequential(
        [
            # Conv1D(16, 3, padding="valid"),
            # Conv1D(16, 3, padding="valid"),
            # BatchNormalization(),
            # Activation("relu"), 
            # Dropout(dropout_rate),

            Conv1D(64, 9, padding="valid"),
            Conv1D(64, 9, padding="valid"),
            BatchNormalization(),
            Activation("relu"), 
            Dropout(dropout_rate),

            Conv1D(128, 9, padding="valid"),
            Conv1D(128, 9, padding="valid"),
            BatchNormalization(),
            Activation("relu"), 
            Dropout(dropout_rate),

            Conv1D(256, 9, padding="valid"),
            Conv1D(256, 9, padding="valid"),
            AveragePooling1D(pool_size=2),
            Dropout(dropout_rate),
            Activation("relu"), 
        ],
        "cnn_block"
    )

    r = cnn_block(real_input)
    i = cnn_block(imag_input)
    sqr = tf.pow(r, 2) - tf.pow(i, 2) + 2 * r * i

    # Concatenate the real and imaginary parts
    layer = Concatenate()([r, i, sqr])
    # layer = Dropout(dropout_rate)(layer)
    layer = Conv1D(1, 32, padding='same', activation='relu')(layer)
    layer = Flatten()(layer)
    # layer = Dense(512, activation="relu")(layer)
    # layer = Dense(256, activation="sigmoid")(layer)
    # layer = Dense(128, activation="sigmoid")(layer)
    # layer = Dense(64, activation="sigmoid")(layer)
    # layer = Dense(32)(layer)
    # layer = Dense(16)(layer)
    # layer = Dense(8)(layer)
    # layer = Dense(4)(layer)
    layer = Dense(1)(layer)

    model = Model(inputs=inputs, outputs=layer, name=name)

    # Allows for us to pass in a complete optimizer or incomplete with learning rate
    if callable(optimizer):
        optimizer = optimizer(learning_rate=initial_learning_rate)

    # finally we compile the model
    # needs to be done in strategy scope
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model


if __name__ == "__main__":
    config_file = sys.argv[1]
    s = SimConfig(config_file)

    MAX_EPOCHS = 10
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
        "initial_learning_rate": 1e-3,
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
        decay_steps=3,
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
        # ModelCheckpoint(
        #     f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
        #     monitor="val_loss",
        #     save_best_only=True,
        #     mode="auto",
        # ),
        TimedLoggingCallback(print_frequency=60),  # custom logger to work a little better with text logs
        # TensorBoard(log_dir=f"{s.tb_dir}/{model_settings['name']}", histogram_freq=1),
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
        input = Input((Alm.getsize(s.lmax), 1, 2), name="complex_split_input")

        opt = Adam(
            learning_rate=lr_schedule,
            # learning_rate=model_settings["initial_learning_rate"],
        )

        model = alm_model(input, opt, metrics, **model_settings)

    # Lets load our data
    tfds_filepath = s.alm_file_complete.replace(".hdf5", "split.tfds")
    data_loader = AlmDataLoader(tfds_filepath, **data_loader_args)
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
        metrics=["loss"] + metrics,
    )
    plot_predictions(y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-preds.png")
    plot_histogram(
        y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-histogram.png"
    )
