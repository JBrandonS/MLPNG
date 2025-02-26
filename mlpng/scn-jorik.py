import os
import sys
import math
import logging
import healpy as hp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NCCL_DEBUG"] = "INFO"

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
    ModelCheckpoint,
)
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
from mlpng.utils.callbacks import LinearWarmup


logger = setup_logging(__name__, level=logging.DEBUG)


def get_model(input_shape, batch_size=32, n_out=1):
    nside = hp.npix2nside(input_shape[1])
    indices = np.arange(input_shape[1])
    layers = []

    n_layers = math.floor(math.log(nside, 2))
    for i in range(n_layers):
        fout = 2 ** (4 + i)
        layers.append(
            HealpyChebyshev(
                K=2,
                Fout=fout,
                use_bias=True,
                use_bn=True,
                activation=LeakyReLU(0.3),
            )
        )
        layers.append(Dropout(0.1))
        layers.append(HealpyPool(1, "AVG"))

    layers.append(Flatten())
    layers.append(Dropout(0.1))
    layers.append(Dense(32, activation=LeakyReLU(0.3)))
    layers.append(Dense(32, activation=LeakyReLU(0.3)))
    layers.append(Dense(n_out))

    model = HealpyGCNN(
        nside,
        indices=indices,
        layers=layers,
        max_batch_size=batch_size,
        initial_Fin=input_shape[-1],
        n_neighbors=8,
    )

    model.build(input_shape)
    return model


def main():
    batch_size = 64
    max_epochs = 100
    shapes = ["local"]

    core = Core()
    ds = MapDataset.fromCore(core, shapes)
    train, val, test = ds.split(
        batch_size=batch_size,
        # cache_file=f"/lustre/smuexa01/client/users/stevensonb/tfcache/{core.name}-{core.slurm.job}.cache",
    )
    decay_steps = len(train)  # len(train) is once per epoch

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        learning_rate = ExponentialDecay(1e-6, decay_steps, 0.95, staircase=True)
        learning_rate = LinearWarmup(learning_rate, decay_steps * 10, 1e-8)

        model = get_model((batch_size, core.npix, core.npols), batch_size, 3)
        model.compile(
            optimizer=AdamW(learning_rate),  # type: ignore
            loss="mse",
            metrics=[RootMeanSquaredError()],  # type: ignore
        )

    model.summary()

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True),
    ]
    if core.use_tb:
        callbacks.append(
            TensorBoard(
                log_dir=core.dirs["tb"],
            )
        )
    if core.use_wandb:
        try_init_wandb(
            config={
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "learning_rate": 5e-3,
            },
            dir=core.dirs["data"],
            append_to=callbacks,
            patch_tb=core.use_tb,
        )

    history = model.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(f"{core.dirs['model']}/{core.name}-{core.slurm.job}-jorik.keras")

    logger.debug("Getting final plots")
    y = np.concatenate([y for _, y in test])  # type: ignore
    preds = model.predict(test, verbose=0)
    # using evaluate to get the rmse since manually calculating it was giving a different value
    rmse = model.evaluate(test, verbose=0)[1]
    fisher = ds.get_fisher("local")

    print_errors(y, preds, fisher)
    plot_metrics(history, metrics=["loss"], save_file=core.get_plot_file("jorik-loss"))
    print(y.shape)
    for shape in range(y.shape[1] or 1):
        plot_predictions(
            y[:, shape],
            preds[:, shape],
            fisher=fisher,
            title=f"RMSE: {rmse:.3f}",
            save_file=core.get_plot_file("jorik-preds"),
        )
        plot_histogram(
            y[:, shape], preds[:, shape], save_file=core.get_plot_file("jorik-hist")
        )


if __name__ == "__main__":
    sys.exit(main())
