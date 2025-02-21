import os
import sys
import math
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import healpy as hp
import numpy as np
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

from mlpng import Core
from mlpng.utils import (
    setup_logging,
    get_fisher,
    plot_predictions,
    plot_histogram,
    print_errors,
    plot_metrics,
    try_init_wandb,
)
from mlpng.utils.dataloaders import tfds_from_hdf5

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
)


logger = setup_logging(__name__, level=logging.DEBUG)
# tf.get_logger().setLevel(logging.ERROR)

# tf.debugging.set_log_device_placement(True)


def SCNJorik(input_shape):
    nside = hp.npix2nside(input_shape[-2])
    indices = np.arange(input_shape[-2])
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
    layers.append(Dropout(0.3))
    layers.append(Dense(32, activation=LeakyReLU(0.3)))
    layers.append(Dense(32, activation=LeakyReLU(0.3)))
    layers.append(Dense(1))

    model = HealpyGCNN(
        nside,
        indices=indices,
        layers=layers,
        n_neighbors=8,
        max_batch_size=input_shape[0],
        initial_Fin=input_shape[-1],
    )

    model.build(input_shape)
    return model


def main():
    core = Core()
    fisher = get_fisher(core.file, "fisher_iso")
    npix = hp.nside2npix(core.nside)
    lensed = False
    lstr = "_lensed" if lensed else ""

    MAX_EPOCHS = 100
    BATCH_SIZE = 32

    train, val, test = tfds_from_hdf5(
        core,
        f"/map{lstr}",
        "/fnl",
        data_fraction=core.data_fraction,
        batch_size=BATCH_SIZE,
        buffer=BATCH_SIZE * 10,
    )

    decay_steps = 1
    logger.debug(
        "length of training data: %s, batch size: %s, decay steps: %s",
        len(train),
        BATCH_SIZE,
        decay_steps,
    )

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        learning_rate = ExponentialDecay(5e-3, decay_steps, 0.95, staircase=True)

        model = SCNJorik((BATCH_SIZE, npix, core.npols))
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
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    if core.use_tb:
        callbacks += [
            TensorBoard(log_dir=core.dirs["tb"]),
            ModelCheckpoint(core.dirs["tb"], monitor="val_loss", save_best_only=True),
        ]
    if core.use_wandb:
        try_init_wandb(
            notes="model testing",
            tags=["SCNJorik"],
            append_to=callbacks,
            patch_tb=core.use_tb,
        )

    history = model.fit(
        train,
        epochs=MAX_EPOCHS,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(f"{core.dirs['model']}/{core.slurm.job}-jorik{lstr}.keras")

    logger.debug("Getting final plots")
    truth = np.concatenate([y for _, y in test])  # type: ignore
    preds = model.predict(test, verbose=0)

    # using evaluate to get the rmse since manually calculating it was giving a different value
    rmse = model.evaluate(test, verbose=0)[1]

    print_errors(truth, preds, fisher)
    plot_metrics(
        history,
        metrics=["loss"],
        save_file=core.get_plot_file(f"jorik_loss{lstr}"),
    )
    plot_predictions(
        truth,
        preds,
        fisher=fisher,
        title=f"RMSE: {rmse:.3f}",
        save_file=core.get_plot_file(f"jorik_preds{lstr}"),
    )
    plot_histogram(
        truth,
        preds,
        save_file=core.get_plot_file(f"jorik_hist{lstr}"),
    )


if __name__ == "__main__":
    tf.profiler.experimental.start("preflogs")
    main()
    tf.profiler.experimental.stop()
