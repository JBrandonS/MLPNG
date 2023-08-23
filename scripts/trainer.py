import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 1
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"

import tensorflow as tf

keras = tf.keras  # fixes issues with vsCode
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})

from keras import Input, Model
from keras.layers import (
    Add,
    Flatten,
    Dense,
    RandomRotation,
    RandomFlip,
)
from keras.optimizers import Adam

from datetime import datetime

from keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from dataloader import DataLoader

from isensee import *
import wandb

from config import SimConfig


def mk_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating dir: {dir}")
        try:
            os.makedirs(dir)
        except FileExistsError:
            # race condition, can happen in job arrays
            pass
    else:
        print(f"Using existing folder: {dir}")


def get_data(start, step, name=None):
    ret = d.skip(start).take(step)

    # This just fixes the logging output not knowing the dataset size
    ret = ret.apply(tf.data.experimental.assert_cardinality(step))

    ret = ret.cache()
    ret = ret.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, name=name)
    ret = ret.prefetch(tf.data.AUTOTUNE)
    return ret


def plt_pred(dataset, name, save=False):
    ipreds = isensee_model.predict(
        dataset, batch_size=s.batch_size, verbose="auto", callbacks=callbacks
    )[:, 0]
    bpreds = bs_model.predict(
        dataset, batch_size=s.batch_size, verbose="auto", callbacks=callbacks
    )[:, 0]

    truth = dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()
    truth = np.array(list(truth))

    irmse = np.sqrt(((ipreds - truth) ** 2).mean())
    brmse = np.sqrt(((bpreds - truth) ** 2).mean())

    plt.plot(truth, ipreds, ".", label=f"isensee: {irmse:.2f}")
    plt.plot(truth, bpreds, ".", label=f"bs: {brmse:.2f}")
    plt.plot(truth, truth, ".", label="truth")
    plt.title(f"{name} predictions")
    plt.xlabel(f"truth")
    plt.ylabel("prediction")

    plt.legend()
    plt.grid()
    plt.show()

    if save:
        plt.savefig(f"{s.plot_dir}/{s.base_name}_{name}.png")


def make_bs_model(
    inputs,
    depth=5,
    dropout_rate=0.3,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="custom_model",
    preprocess=False,
):
    if preprocess:
        inputs = RandomFlip("horizontal")(inputs)
        # inputs = RandomZoom(0.2)(inputs)
        inputs = RandomRotation(np.pi)(inputs)

    current_layer = inputs
    level_output_layers = []
    level_filters = []
    n_level_filters = 16
    for _ in range(depth):
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters, strides=(2, 2))

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate
        )
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    flat_layer = Flatten()(current_layer)
    out_layer = Dense(1, activation=None)(flat_layer)

    model = Model(inputs=inputs, outputs=out_layer, name=name)
    model.compile(
        optimizer=optimizer(learning_rate=initial_learning_rate),
        loss=loss_function,
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.KLDivergence(),
        ],
    )
    return model


if __name__ == "__main__":
    print(f"tf version: {tf.__version__}")

    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings/settings.json"
    s = SimConfig(config_file)

    batch_size = 32  # TODO auto find optimal batch_size based on nside
    max_epochs = 100

    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") # for debugging

    # holds base settings for wandb
    wandb_config = s.settings

    mk_dir(s.data_dir)
    mk_dir(s.model_dir)
    mk_dir(s.plot_dir)
    mk_dir(s.tb_dir)

    n = s.total_sims
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = int(n * 0.1)

    # load data as a generator so we do not need to have it all in memory
    d = tf.data.Dataset.from_generator(
        DataLoader(s.data_file_complete),
        output_signature=(
            tf.TensorSpec(shape=(s.nside, s.nside), dtype=s.r_dtype),
            tf.TensorSpec(shape=(), dtype=s.r_dtype),
        ),
    )

    # fix a log warning
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    d = d.with_options(options)

    train_dataset = get_data(0, train_size, "train")
    val_dataset = get_data(train_size, val_size, "val")
    test_dataset = get_data(train_size + val_size, test_size, "test")

    with strategy.scope():
        model_settings = {
            "depth": 4,
            "n_segmentation_levels": 4,
            "dropout_rate": 0.03,
            "loss_function": tf.keras.losses.mse,
            "initial_learning_rate": 0.01,
            "name": f"isensee-{s.base_name}",
        }

        wandb.init(
            project="mlpng",
            notes="isensee",
            tags=["isensee", "dev"],
            reinit=True,
            tensorboard=True,
            #    sync_tensorboard=True,
            config=wandb_config | model_settings,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_root_mean_square_error",
                patience=40,
                verbose=1,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_root_mean_squared_error", factor=0.1, patience=10
            ),
            WandbMetricsLogger(),
            WandbModelCheckpoint(filepath=f"{s.model_dir}/wandb"),
            TensorBoard(log_dir=s.tb_dir),
        ]

        input_img = Input((s.nside, s.nside, 1), name="img")

        isensee_model = isensee2017_model(input_img, **model_settings)
        if s.debug and s.verbose:
            isensee_model.summary()

        isensee_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=max_epochs,
            callbacks=callbacks,
        )

        isensee_model.predict(
            test_dataset,
            batch_size=batch_size,
            verbose="auto",
            callbacks=callbacks,
            use_multiprocessing=True,
        )

        wandb.finish(0)

    with strategy.scope():
        model_settings = {
            "depth": 4,
            "dropout_rate": 0.3,
            "loss_function": tf.keras.losses.mse,
            "initial_learning_rate": 0.01,
            "preprocess": False,
            "name": f"bs-{s.base_name}",
        }

        wandb.init(
            project="mlpng",
            notes="bs",
            tags=["bs", "dev"],
            reinit=True,
            config=wandb_config | model_settings,
        )

        bs_model = make_bs_model(input_img, **model_settings)
        if s.debug and s.verbose:
            bs_model.summary()

        bs_model.fit(
            train_dataset,
            validation_data=val_dataset,
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=callbacks,
        )

        bs_model.predict(
            test_dataset, batch_size=batch_size, verbose="auto", callbacks=callbacks
        )
