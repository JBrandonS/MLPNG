import numpy as np
import keras_tuner as kt
import tensorflow as tf
import healpy
import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

from deepsphere import HealpyGCNN, healpy_layers as hp_layer
from deepsphere.healpy_layers import HealpyChebyshev, HealpyPool

from mlpng import Core
from mlpng.utils.dataloaders import HDF5Dataset


def model_builder(hp):
    K = hp.Choice("K", [5, 10, 11, 17])
    first_bn = hp.Boolean("first_bn")
    inter_bn = hp.Boolean("inter_bn")
    inter_act = hp.Choice("inter_act", ["relu", "linear"])
    pt1 = hp.Choice("pool_type_1", ["MAX", "AVG"])
    layers = [
        HealpyChebyshev(
            K=K, Fout=32, use_bias=True, use_bn=first_bn, activation="relu"
        ),
        HealpyChebyshev(
            K=K, Fout=32, use_bias=True, use_bn=inter_bn, activation=inter_act
        ),
        HealpyPool(p=1, pool_type=pt1),
        HealpyChebyshev(K=K, Fout=64, use_bias=True, use_bn=True, activation="relu"),
        HealpyChebyshev(
            K=K, Fout=64, use_bias=True, use_bn=inter_bn, activation=inter_act
        ),
        HealpyPool(p=1, pool_type=pt1),
        HealpyChebyshev(K=K, Fout=128, use_bias=True, use_bn=True, activation="relu"),
        HealpyChebyshev(
            K=K, Fout=128, use_bias=True, use_bn=inter_bn, activation=inter_act
        ),
        HealpyPool(p=1, pool_type=pt1),
        HealpyChebyshev(K=K, Fout=256, use_bias=True, use_bn=True, activation="relu"),
        HealpyChebyshev(
            K=K, Fout=256, use_bias=True, use_bn=inter_bn, activation=inter_act
        ),
        HealpyPool(p=1, pool_type=hp.Choice("pool_type_4", ["MAX", "AVG"])),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),  # hp.Choice("dropout", [0.0, 0.3, 0.5])),
        keras.layers.Dense(
            128, "relu"
        ),  # hp.Choice("dense_1", [64, 128, 256, 512]), "relu"),
        keras.layers.Dense(64),  # hp.Choice("dense_2", [8, 16, 32, 64])),
        keras.layers.Dense(1),
    ]

    indices = np.arange(healpy.nside2npix(core.nside))
    model = HealpyGCNN(
        nside=core.nside,
        indices=indices,
        layers=layers,
        n_neighbors=8,
    )
    model.build(input_shape=(None, len(indices), 1))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=100,
        decay_rate=0.95,
        staircase=True,
    )

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=keras.losses.mse)
    return model  # .keras_model()


if __name__ == "__main__":
    args = [
        "settings/n128.json",
        "--nsims",
        "100",
        "--narray",
        "1000",
        "--pols",
        "T",
        "--fnl_range",
        "-100",
        "100",
    ]
    core = Core(args)
    indices = np.arange(healpy.nside2npix(core.nside))

    def to_tf(ds):
        return (
            tf.data.Dataset.from_generator(
                lambda: ds,
                output_signature=(
                    tf.TensorSpec(shape=(len(indices), 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                ),
            )
            .apply(tf.data.experimental.assert_cardinality(len(ds)))
            .cache()
            .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
            .batch(
                16,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    ds = HDF5Dataset(core.file, x_name="alm", y_name="fnl")
    f = 100
    train_ds, test_ds, val_ds = ds.split(0.8 / f, 0.1 / f, 0.1 / f)
    train_tf, test_tf, val_tf = map(to_tf, (train_ds, test_ds, val_ds))

    gpus = tf.config.list_physical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy([f"/gpu:{i}" for i in range(len(gpus))])

    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        executions_per_trial=3,
        directory="data/tuning",
        project_name="deepsphere_v2",
        distribution_strategy=strategy,
    )
    tuner.search_space_summary(True)

    tuner.search(
        train_tf,
        validation_data=val_tf,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5),
            keras.callbacks.TensorBoard(f"{core.dirs['base']}/tb/deepsphere-tuning"),
        ],
    )

    tuner.results_summary()
