import os
import sys
import math
import logging
import healpy as hp
import numpy as np

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    TensorBoard,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
)

from mlpng import Core
from mlpng.utils import (
    setup_logging,
    try_init_wandb,
    make_trainer_plots,
)
from mlpng.utils.dataloaders import MapDataset
from mlpng.utils.callbacks import RMSELoss, rmse_metrics


logger = setup_logging(__name__, level=logging.DEBUG)


def get_model(input_shape, max_batch_size=32, n_out=1):
    """This is the jorik model from the paper, add links to the paper and code if available."""
    nside = hp.npix2nside(input_shape[1])
    layers = []

    n_layers = math.floor(math.log(nside, 2))
    for i in range(n_layers):
        fout = 32
        layers.append(
            HealpyChebyshev(
                K=2,
                Fout=fout,  # this is not mentioned in paper
                # use_bias=True,  # this is not mentioned in paper
                use_bn=True,  # this is not mentioned in paper
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
    lensed = True
    run_name = "jorik-818"

    core = Core()
    shapes = core.shapes

    # here we grab out dataset, splitting and duplicating based on paper
    ds = MapDataset.fromCore(core, lensed=lensed)
    train, val, test = ds.split(
        train_size=0.4,
        val_size=0.1,
        test_size=0.5,
        to_tf=True,
        batch_size=batch_size,
        duplicates=[25, 10, 2],
        cache_file=f"{core.name}-050625-{run_name}-{core.shapes_str()}",
        gen_batch_size=16,
    )

    # the paper does not rotate the test set
    test.rotate = False

    # calculate the number of steps per epoch and decay steps for use later
    epoch_steps = math.ceil(core.total_sims * 0.4 * 25 // batch_size)
    decay_steps = epoch_steps * 1

    # get this data that we will need later, also serves to create the test data
    # this allows us to have the full cached dataset by the end of the first epoch
    # if the cache exists this is very fast, might need to change this for very large datasets
    logger.debug("Creating test cache")
    y_test = np.concatenate([y for _, y in test])
    logger.debug("Finished creating test cache")

    strategy = tf.distribute.MirroredStrategy()  # use mirrored strategy for multi-GPU
    with strategy.scope():
        learning_rate = 5e-4
        learning_rate = ExponentialDecay(
            learning_rate, decay_steps, 0.95, staircase=True
        )

        model = get_model((None, core.npix, core.npols), batch_size, len(shapes))

        model.compile(
            optimizer=AdamW(learning_rate),
            loss=RMSELoss(),
            metrics=rmse_metrics(shapes),
        )

    model.summary()

    # create the callbacks
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True),
    ]

    # if wanted we create a tensorboard and wandb callback, use the CLI to set these
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

    # and finally we fit the model
    history = model.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )

    model_file = f"{core.dirs['model']}/{core.name}-{run_name}-{core.slurm.job}.keras"
    logger.info("Saving model to %s", model_file)
    model.save(model_file)

    if core.plot:
        logger.debug("Getting final plots")
        preds = model.predict(test, verbose=0)
        metrics = model.evaluate(test, verbose=0)
        plot_dir = os.path.join(core.dirs["plot"], str(core.name), run_name)

        make_trainer_plots(
            core,
            plot_dir,
            run_name,
            history,
            metrics,
            y_test,
            preds,
            core.get_likelihoods(lensed),
        )


if __name__ == "__main__":
    logger.info("Conda environment: %s", os.environ["CONDA_DEFAULT_ENV"])
    logger.info("Python executable: %s", sys.executable)
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("CUDA version: %s", tf.sysconfig.get_build_info()["cuda_version"])
    logger.info("cuDNN version: %s", tf.sysconfig.get_build_info()["cudnn_version"])

    sys.exit(main())
