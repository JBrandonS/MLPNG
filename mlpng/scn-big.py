import os
import sys
import math
import logging
import healpy as hp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import (  # type: ignore
    Dense,
    Dropout,
    Flatten,
    ReLU,
    LeakyReLU,
    AlphaDropout,
)
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping,
    TerminateOnNaN,
    TensorBoard,
)
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts  # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError  # type: ignore
from tensorflow.keras.initializers import LecunNormal

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

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

logger = setup_logging("mlpng.scn_shapes", level=logging.DEBUG)
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
        # fout = 32  # 128
        fout = 2 ** (6 + i)
        layers.append(
            HealpyChebyshev(
                K=2,
                Fout=fout,
                # use_bias=True,
                use_bn=True,
                activation="relu",
            )
        )
        layers.append(
            HealpyChebyshev(
                K=2,
                Fout=fout,
                # use_bias=True,
                use_bn=True,
                activation="relu",
            )
        )
        # layers.append(Dropout(0.1))
        layers.append(HealpyPool(1, "AVG"))

    layers.append(Flatten())
    layers.append(Dropout(0.3))
    layers.append(Dense(512, activation="relu"))
    layers.append(Dense(256))
    layers.append(Dense(128, activation="relu"))
    layers.append(Dense(64))
    layers.append(Dense(32, activation="relu"))
    layers.append(Dense(16))
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


def weighted_mse(sigmas, index=None, name=None):
    sigma = tf.constant(sigmas, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        if index is None:
            return tf.reduce_mean(tf.square((y_true - y_pred) / sigma), axis=0)
        else:
            # only compute the loss for the specified index
            error = (y_true[:, index] - y_pred[:, index]) / sigma[index]
            return tf.reduce_mean(tf.square(error))

    # set the name for the metric if provided
    loss_fn.__name__ = f"wmse_{name}" if name else "weighted_mse"

    # return a tf.function now for improvements
    # cannot use the @tf.function decorator here because it will not work with the name change
    return tf.function(loss_fn)


def wmse_metric(sigmas, shapes):
    """
    Create a metric function that computes the weighted mean squared error
    for each shape in the list of shapes.
    """
    sigmas = np.array(sigmas, dtype=np.float32)
    metrics = []
    for i, shape in enumerate(shapes):
        metrics.append(weighted_mse(sigmas, i, shape))
    return metrics


def imse(index=None, name=None):
    def loss_fn(y_true, y_pred):
        if index is None:
            return tf.reduce_mean(tf.square((y_true - y_pred)), axis=0)
        else:
            # only compute the loss for the specified index
            error = y_true[:, index] - y_pred[:, index]
            return tf.reduce_mean(tf.square(error))

    # set the name for the metric if provided
    loss_fn.__name__ = f"mse_{name}" if name else "mse"

    # return a tf.function now for improvements
    # cannot use the @tf.function decorator here because it will not work with the name change
    return tf.function(loss_fn)


def imse_metric(shapes):
    metrics = []
    for i, shape in enumerate(shapes):
        metrics.append(imse(i, shape))
    return metrics


def irmse(index=None, name=None):
    def loss_fn(y_true, y_pred):
        if index is None:
            return tf.sqrt(tf.reduce_mean(tf.square((y_true - y_pred))))  # , axis=0))
        else:
            # only compute the loss for the specified index
            error = y_true[:, index] - y_pred[:, index]
            return tf.sqrt(tf.reduce_mean(tf.square(error)))

    # set the name for the metric if provided
    loss_fn.__name__ = f"rmse_{name}" if name else "rmse"

    # return a tf.function now for improvements
    # cannot use the @tf.function decorator here because it will not work with the name change
    return tf.function(loss_fn)


def irmse_metric(shapes):
    metrics = []
    for i, shape in enumerate(shapes):
        metrics.append(irmse(i, shape))
    return metrics


def shapes_str(shapes):
    if isinstance(shapes, str):
        shapes = [shapes]  # Convert to a list for consistency
    if len(shapes) == 1:
        return shapes[0]
    return "".join([shape[0] for shape in shapes])


def main():
    batch_size = 128
    max_epochs = 300

    core = Core()
    shapes = core.shapes

    ds = MapDataset.fromCore(core)
    train, val, test = ds.split(
        0.8,
        0.1,
        0.1,
        to_tf=True,
        batch_size=batch_size,
        duplicates=[25, 10, 5],
        cache_file=f"{core.name}-032925-{shapes_str(shapes)}",
        gen_batch_size=16,
    )
    # the paper does not rotate the test set
    test.rotate = False  # type: ignore

    epoch_steps = math.ceil(core.total_sims * 0.4 * 25 // batch_size)
    decay_steps = epoch_steps * 2

    fishers = ds.get_fisher(shapes)
    sigmas = np.sqrt(1 / fishers)
    logger.debug("fishers: %s, std: %s", fishers, sigmas)

    # get this data that we will need later, also serves to create the test data
    # this allows us to have the full cached dataset by the end of the first epoch
    # if the cache exists this is very fast
    logger.debug("Creating test cache")
    y_test = np.concatenate([y for _, y in test])  # type: ignore
    logger.debug("Done creating test cache")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        learning_rate = 5e-4
        # learning_rate = ExponentialDecay(
        #     learning_rate, decay_steps, 0.9, staircase=True
        # )
        learning_rate = CosineDecayRestarts(
            learning_rate, decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.01
        )

        model = get_model((None, core.npix, core.npols), batch_size, len(shapes))
        model.compile(
            optimizer=AdamW(learning_rate, weight_decay=0.0001),  # type: ignore
            loss=irmse(),
            metrics=irmse_metric(shapes),  # type: ignore
        )

    model.summary()

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
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
    rmse = model.evaluate(test, verbose=2)

    model_file = f"{core.dirs['model']}/{core.name}-{core.slurm.job}.keras"
    logger.info("Saving model to %s", model_file)
    model.save(model_file)

    # print_errors(y_test, preds, fishers)
    plot_metrics(
        history,
        metrics=["loss"] + [f"rmse_{s}" for s in shapes],
        save_file=core.get_plot_file("loss"),
    )
    for s in range(y_test.shape[1]):
        s_str = shapes[s]
        plot_predictions(
            y_test[:, s],
            preds[:, s],
            fisher=fishers[s],  # type: ignore
            title=f"{s_str} RMSE: {rmse[1 + s]:.3f}",
            save_file=core.get_plot_file(f"preds-{s_str}"),
        )
        # plot_histogram(
        #     y_test[:, s],
        #     preds[:, s],
        #     title=f"{s_str} RMSE: {rmse[s]:.3f}",
        #     save_file=core.get_plot_file(f"{s_str}-hist"),
        # )


if __name__ == "__main__":
    sys.exit(main())
