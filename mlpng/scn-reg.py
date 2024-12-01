import gc
import os
import sys
import time
import logging
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.layers.convolutional import Conv1D

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import keras
from keras.layers import Layer
from keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    TensorBoard,
    ModelCheckpoint,
)
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay, LearningRateSchedule
from keras.metrics import RootMeanSquaredError

from mlpng import Core
from mlpng.utils import (
    setup_logging,
    load_data,
    get_fisher,
    plot_predictions,
    plot_histogram,
    print_errors,
    plot_metrics,
    WarmupLearningRate,
    AttentionSchedule,
    try_init_wandb,
)
from mlpng.utils.dataloaders import HDF5Dataset

from deepsphere import HealpyGCNN, healpy_layers as hp_layer
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
    HealpyPseudoConv_Transpose,
)
from deepsphere.gnn_layers import Chebyshev
import tfkan
import keract


logger = setup_logging(__name__, level=logging.DEBUG)

MAX_EPOCHS = 100
BATCH_SIZE = 8


class SCNBlock(keras.Model):

    def __init__(
        self,
        filters,
        nside=128,
        n_mid=None,
        n_out=None,
        n_neighbors=20,
        max_batch_size=BATCH_SIZE,
        cheb_degree=11,
        cheb_init=None,
        cheb_act=None,
        cheb_bias=True,
        cheb_batch=True,
        cheb_bnl=None,
        cheb_splits=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.n_neighbors = n_neighbors
        self.max_batch_size = max_batch_size
        if n_mid is None:
            n_mid = filters * 4
        if n_out is None:
            n_out = filters
        self.n_mid = n_mid
        self.n_out = n_out
        self.cheb_degree = cheb_degree
        self.cheb_init = cheb_init
        self.cheb_act = cheb_act
        self.cheb_bias = cheb_bias
        self.cheb_batch = cheb_batch
        self.cheb_splits = cheb_splits

        self.lin_norm = keras.layers.LayerNormalization()  # type: ignore
        self.act = keras.layers.Activation("gelu")
        self.conv1 = HealpyChebyshev(
            K=cheb_degree,
            initializer=cheb_init,
            activation=cheb_act,
            use_bias=cheb_bias,
            use_bn=cheb_batch,
            depth_wise=True,
        )
        self.conv2_1d = keras.layers.Conv1D(n_mid, kernel_size=1)
        self.conv3_1d = keras.layers.Conv1D(n_out, kernel_size=1)
        self.res_conv_1d = keras.layers.Conv1D(filters, kernel_size=1)
        self.gcnn = HealpyGCNN(
            int(nside),
            np.arange(hp.nside2npix(int(nside))),
            layers=[
                self.conv1,
                self.lin_norm,
                self.conv2_1d,
                self.act,
                # self.grn,
                self.conv3_1d,
            ],
            verbose=True,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.gcnn.build(input_shape)

    def call(self, inputs, training=None, mask=None):
        res = self.res_conv_1d(inputs)
        x = self.gcnn(inputs)
        return x + res


def SphereConvNeXt(
    input_shape,
    nside=128,
    base_channels=32,
    n_neighbors=8,
    blocks=[
        1,
        1,
        3,
        1,
    ],
):
    base_channels = base_channels
    n_neighbors = n_neighbors
    blocks = blocks
    channels = [base_channels * 2**p for p in range(len(blocks))]

    layers = []
    for block in range(len(blocks)):
        filters = channels[block]
        for _ in range(blocks[block]):
            layers.append(SCNBlock(filters, nside))
        layers.append(HealpyPool(1))
        nside /= 2

    final_layers = [
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.LayerNormalization(),
        keras.layers.Dense(128, activation="gelu"),
        keras.layers.Dense(256),
        keras.layers.Dense(512),
        keras.layers.Dense(128, activation="gelu"),
        keras.layers.Dense(32),
        keras.layers.Dense(1),
    ]

    [layers.append(layer) for layer in final_layers]
    return keras.models.Sequential(layers)


class SphereUNet(keras.Model):
    def __init__(
        self,
        nside=128,
        base_channels=32,
        n_neighbors=20,
        blocks=[1, 1, 3, 1],
        pools=[1, 1, 1, 1],
        **kwargs,
    ):
        super(SphereUNet, self).__init__(**kwargs)
        self.base_channels = base_channels
        self.n_neighbors = n_neighbors
        self.blocks = blocks
        self.pools = pools

        self.channels = [base_channels * 2**p for p in range(len(blocks))]
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.pool_layers = []
        self.up_layers = []

        for block in range(len(blocks)):
            encoder_block = []
            for _ in range(blocks[block]):
                encoder_block.append(
                    SCNBlock(
                        self.channels[block],
                        nside=nside,
                        cheb_degree=7,
                        n_neighbors=n_neighbors,
                        cheb_bias=True,
                    )
                )
            self.encoder_blocks.append(encoder_block)
            self.pool_layers.append(HealpyPool(p=pools[block], pool_type="MAX"))
            nside /= 2

        nside *= 2

        for block in reversed(range(len(blocks))):
            decoder_block = []
            for _ in range(blocks[block]):
                decoder_block.append(
                    SCNBlock(
                        self.channels[block],
                        nside=nside,
                        cheb_degree=7,
                        n_neighbors=n_neighbors,
                        cheb_bias=True,
                    )
                )
            self.decoder_blocks.append(decoder_block)
            self.up_layers.append(
                HealpyPseudoConv_Transpose(pools[block], self.channels[block])
            )
            nside *= 2

    def call(self, inputs, training=None, mask=None):
        x = inputs
        skips = []

        # Encoder
        for encoder_block, pool_layer in zip(self.encoder_blocks, self.pool_layers):
            for layer in encoder_block:
                x = layer(x)
            skips.append(x)
            x = pool_layer(x)

        # Decoder
        for decoder_block, up_layer, skip in zip(
            self.decoder_blocks, self.up_layers, reversed(skips)
        ):
            x = up_layer(x)
            x = keras.layers.Concatenate()([x, skip])
            for layer in decoder_block:
                x = layer(x)

        x = Conv1D(1, 1)(x)
        return x


def get_model(nside=128):
    full_model = keras.models.load_model(saved_model_path)
    full_model.summary()

    # Extract encoder layers
    encoder_input = full_model.input
    encoder_output = None
    for layer in full_model.layers:
        if (
            "scn_block_6" in layer.name
        ):  # Assuming pooling layers mark the end of encoder blocks
            encoder_output = layer.output
            break

    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_output)

    # Add MLP layers on top of the encoder
    mlp_input = encoder_model.output
    x = keras.layers.Flatten()(mlp_input)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(1)(x)  # Output scalar value
    return keras.Model(inputs=encoder_input, outputs=x)


def to_tf(ds, npix):
    return (
        tf.data.Dataset.from_generator(
            lambda: ds,
            output_signature=(
                tf.TensorSpec(shape=(npix, 1), dtype=tf.float32),  # type: ignore
                tf.TensorSpec(shape=(), dtype=tf.float32),  # type: ignore
            ),
        )
        # .map(lambda x, y: (x, x))  # for unet
        .apply(tf.data.experimental.assert_cardinality(len(ds)))
        .cache()
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .batch(
            BATCH_SIZE,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def main():
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
    fisher = get_fisher(core.file)
    npix = hp.nside2npix(core.nside)

    f = 100
    ds = HDF5Dataset(core.file, x_name="alm", y_name="fnl")
    train_ds, test_ds, val_ds = ds.split(0.7 / f, 0.2 / f, 0.1 / f, verbose=True)
    train_tf, test_tf, val_tf = (to_tf(x, npix) for x in [train_ds, test_ds, val_ds])

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        decay_steps = len(train_ds) // BATCH_SIZE  # once per epoch
        learning_rate = ExponentialDecay(4e-3, decay_steps, 0.95, staircase=True)
        # learning_rate = keras.optimizers.schedules.CosineDecay(4e-3, 20)
        metrics = []  # root_mean_squared_error, "mse"]

        hloss = keras.losses.Huber(delta=1 / np.sqrt(fisher))

        # model = SphereConvNeXt((BATCH_SIZE, npix, 1))
        model = get_model()
        model.build((BATCH_SIZE, npix, 1))
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate,  # weight_decay=0.05, use_ema=True, ema_momentum=0.9999
            ),
            loss="mse",  # root_mean_squared_error,
            metrics=metrics,  # type: ignore
        )

    model.summary()

    tf_dir = f"{core.dirs['tb']}/{core.name}"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        TensorBoard(log_dir=tf_dir),
        TerminateOnNaN(),
        ModelCheckpoint(tf_dir, monitor="val_loss", save_best_only=True),
    ]
    try_init_wandb(notes="model testing", tags=["SCN"], append_to=callbacks)

    history = model.fit(
        train_tf,
        epochs=MAX_EPOCHS,
        validation_data=val_tf,
        callbacks=callbacks,
        verbose=1,
    )

    model.evaluate(test_tf, verbose=2)  # type: ignore
    preds = model.predict(test_tf, verbose=2).flatten()  # type: ignore
    truth = np.concatenate([y for _, y in test_tf])  # type: ignore

    name = "SCNU_2"
    save_base = core.dirs["plot"] + f"/SCN-{core.name}"
    os.makedirs(save_base, exist_ok=True)
    print_errors(truth, preds, fisher)
    plot_metrics(history, metrics=["loss"], save_file=save_base + f"/{name}-loss.png")
    plot_predictions(
        truth,
        preds,
        fisher=fisher,
        save_file=save_base + f"/{name}-preds.png",
    )
    plot_histogram(truth, preds, save_file=save_base + f"/{name}-hist.png")

    # first_batch = next(iter(test_tf.take(1)))
    # data, _ = first_batch
    # activations = keract.get_activations(model, data, auto_compile=True)
    # print(activations.keys(), flush=True)
    # # TODO for each activation layer read in the weights and use
    # # hp.mollview(original_map[0,:,0], nest=True, title="Input Map")
    # # activations = activations.get(layers)
    # keract.display_activations(
    #     activations,
    #     save=True,
    #     directory=os.path.join(core.dirs["plot"], f"{core.name}-activations"),
    #     data_format="channels_last",
    # )


if __name__ == "__main__":
    sys.exit(main())
