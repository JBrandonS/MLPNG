import sys
import json
import logging
import os
import time

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN,
    TensorBoard,
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import mse

from .models import AutoModel
from .utils import setup_logging, log_source, try_init_wandb, get_fisher
from .utils.plots import plot_histogram, plot_predictions
from .utils.tf.plots import plot_metrics
from .utils.tf.callbacks import (
    TimedLoggingCallback,
    WarmupLearningRate,
    AttentionSchedule,
)

logger = setup_logging("trainer", logging.DEBUG)  # dont want name to be __main__


def main():
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    logger.info(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")

    # here we get our model using the CLI args with --model provided
    model = AutoModel()
    model_cls = model.__class__.__name__

    # These settings, esp the batch size, should be set by the model so we do
    MAX_EPOCHS = 300
    BATCH_SIZE = model.BATCH_SIZE

    run_start_time = int(time.time())

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": model.sjob,
        "start_time": run_start_time,
        "max_epochs": MAX_EPOCHS,
        "comment": """...""",
    }
    logger.debug(f"Extra info:\n{json.dumps(extra_info, indent=2)}")

    # settings that get passed into the model
    model_settings = {
        "name": f"{extra_info['slurm_job_id']}_{model_cls}_{model.base_name}_{run_start_time}",
    }
    logger.debug(f"Model settings:\n{json.dumps(model_settings, indent=2)}")

    # Might want to change this to be per model since some models might need different settings
    data_settings = {
        "shuffle": False,
        "seed": None,
        "batch_size": BATCH_SIZE,
        "cache": True,
        "shuffle_buffer": 1000,
        "normalize": False,
    }
    logger.debug(f"Data loader settings:\n{json.dumps(data_settings, indent=2)}")

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error"]
    logger.debug(f"Looking at additional metrics: {metrics}")

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=40,
        ),
        # model checkpointing to save the best model
        ModelCheckpoint(
            f"{model.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
            initial_value_threshold=40000,  # mse, only want to bother saving decent models
        ),
        # TimedLoggingCallback(print_frequency=15),  # custom logger to work a little better with text logs
        # TensorBoard(log_dir=f"{core.tb_dir}"),
        TerminateOnNaN(),
    ]

    # enable wandb, if using
    if False:
        try_init_wandb(
            notes=extra_info["comment"],
            tags=[model_cls],
            config={**model_settings, **data_settings},
            append_to=callbacks,
        )

    # lr_schedule = AttentionSchedule(model.lmax)
    # lr_schedule = WarmupLearningRate(warmup_steps=1000)
    lr_schedule = ExponentialDecay(1e-5, 10000, 0.96)

    # get the dataset from the model, also sets the internal dataset for the model
    dataset = model.init_dataset(**data_settings)
    # split the dataset used for the model into train, test, and validation
    train_ds, test_ds, val_ds = dataset.get_split(0.8, 0.1, 0.1)

    # create and compile the model, needs to be in scope of the strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        opt = Adam(learning_rate=lr_schedule)
        model.make_model(**model_settings)
        model.compile(optimizer=opt, loss="mse", metrics=metrics)
        model.summary()

    # Finally, lets fit our model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=2,  # since we are using the custom logger
    )

    # Lets plot the predictions from the unseen test set
    y_pred = model.predict(test_ds, verbose=2).flatten()
    y_test = np.concatenate([y.numpy() for _, y in test_ds])

    # Plot the loss curves and metrics
    plot_dir = os.path.join(model.plot_dir, "trainer")
    file_base = os.path.join(plot_dir, model_settings["name"])
    os.makedirs(plot_dir, exist_ok=True)

    fisher = get_fisher(model.alm_file)
    plot_metrics(history, save_file=f"{file_base}.png", metrics=["loss"] + metrics)
    plot_predictions(y_test, y_pred, fisher=fisher, save_file=f"{file_base}-preds.png")
    plot_histogram(y_test, y_pred, save_file=f"{file_base}-hist.png")


if __name__ == "__main__":
    sys.exit(main())
