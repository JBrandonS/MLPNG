import os
import sys
import math
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import healpy as hp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import (  # type: ignore
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
)
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping,
    TerminateOnNaN,
    TensorBoard,
)
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError  # type: ignore

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
)

from mlpng import Core
from mlpng.utils import (
    setup_logging,
    plot_predictions,
    plot_histogram,
    print_errors,
    plot_metrics,
    try_init_wandb,
)
from mlpng.utils.dataloaders import MapDataset


logger = setup_logging(__name__, level=logging.DEBUG)
logger.info("Conda environment: %s", os.environ["CONDA_DEFAULT_ENV"])
logger.info("Python executable: %s", sys.executable)
logger.info("TensorFlow version: %s", tf.__version__)
logger.info("CUDA version: %s", tf.sysconfig.get_build_info()["cuda_version"])
logger.info("cuDNN version: %s", tf.sysconfig.get_build_info()["cudnn_version"])


def get_model(input_shape, max_batch_size=32, n_out=1):
    nside = hp.npix2nside(input_shape[1])
    layers = []

    n_layers = math.floor(math.log(nside, 2))
    for i in range(n_layers):
        fout = 32
        layers.append(
            HealpyChebyshev(
                K=2,
                Fout=fout,  # this is not mentioned in paper
                # use_bias=True,
                use_bn=True,
                activation=LeakyReLU(0.3),
            )
        )
        layers.append(Dropout(0.1))
        layers.append(HealpyPool(1, "AVG"))

    layers.append(Flatten())
    layers.append(Dropout(0.3))
    layers.append(Dense(32, activation=LeakyReLU(0.3)))
    layers.append(Dense(32, activation=LeakyReLU(0.3)))
    layers.append(Dense(n_out))

    model = HealpyGCNN(
        nside,
        indices=np.arange(input_shape[1]),
        layers=layers,
        n_neighbors=8,
        max_batch_size=max_batch_size,
        initial_Fin=input_shape[-1],
    )

    model.build(input_shape)
    return model


def main():
    batch_size = 64
    max_epochs = 300
    shapes = ["local"]

    core = Core()
    ds = MapDataset.fromCore(core, shapes=shapes)
    # ds = elsnerMapDataset.fromCore(core, shapes=shapes)

    # get our dataset, right now we manually convert to tf to match the
    train, val, test = ds.split(
        0.4,
        0.1,
        0.5,
        to_tf=True,
        batch_size=batch_size,
        duplicates=[25, 10, 2],
        cache_file=core.name,
    )

    # the paper does not rotate the test set
    test.rotate = False  # type: ignore

    epoch_steps = math.ceil(core.total_sims * 0.4 * 25 // batch_size)
    decay_steps = epoch_steps * 1

    # get this data that we will need later, also serves to create the test data
    # this allows us to have the full cached dataset by the end of the first epoch
    # if the cache exists this is very fast
    logger.debug("Creating test cache")
    y_test = np.concatenate([y for _, y in test])  # type: ignore

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        learning_rate = 5e-4
        learning_rate = ExponentialDecay(
            learning_rate, decay_steps, 0.95, staircase=True
        )

        model = get_model((None, core.npix, core.npols), batch_size, len(shapes))
        model.compile(
            optimizer=AdamW(
                learning_rate,  # weight_decay=0.01, global_clipnorm=1.0
            ),  # type: ignore
            loss="mse",
            metrics=[RootMeanSquaredError()],  # type: ignore
        )

    model.summary()

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True),
    ]
    tb_dir = f"{core.dirs['tb']}/{core.name}/{core.slurm.job}"
    if core.use_tb:
        callbacks.append(
            TensorBoard(
                log_dir=tb_dir,
                histogram_freq=1,
                write_steps_per_second=True,
            )
        )
    if core.use_wandb:
        try_init_wandb(
            config={
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "shapes": shapes,
                "learning_rate": repr(learning_rate),
            },
            dir=core.dirs["wandb"],
            append_to=callbacks,
            patch_tb=core.use_tb,
            patch_logdir=tb_dir,
        )

    history = model.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )

    logger.debug("Getting final plots")
    preds = model.predict(test, verbose=0)
    # using evaluate to get the rmse since manually calculating it was giving a different value
    rmse = model.evaluate(test, verbose=0)[1]
    fisher = ds.get_fisher("local")

    # save only if our results are good enough
    if rmse <= 3 * np.sqrt(1 / fisher):  # type: ignore
        model.save(
            f"{core.dirs['model']}/{core.name}-{core.slurm.job}-{rmse:.2f}-jorik.keras"
        )

    print_errors(y_test, preds, fisher)
    plot_metrics(history, metrics=["loss"], save_file=core.get_plot_file("jorik-loss"))
    for shape in range(y_test.shape[1]):
        plot_predictions(
            y_test[:, shape],
            preds[:, shape],
            fisher=fisher,  # type: ignore
            title=f"RMSE: {rmse:.3f}",
            save_file=core.get_plot_file("jorik-preds"),
        )
        plot_histogram(
            y_test[:, shape],
            preds[:, shape],
            save_file=core.get_plot_file("jorik-hist"),
        )


if __name__ == "__main__":
    tf.profiler.experimental.start("preflogs")
    main()
    tf.profiler.experimental.stop()
