import os
import sys
import logging
import healpy as hp
import numpy as np

# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import matplotlib.pyplot as plt
from scipy import rand
import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv1D,
    Dense,
    Dropout,
    LayerNormalization,
    Activation,
    Concatenate,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    TensorBoard,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import AdamW, Adam  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay, LearningRateSchedule  # type: ignore
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


def rotate_ds(inputs, y):
    @tf.py_function(Tout=tf.float32)  # type: ignore
    def rotate_map(inputs):
        lat_angle = tf.random.uniform([1], -90.0, 90.0)  # type: ignore
        lon_angle = tf.random.uniform([1], -180.0, 180.0)  # type: ignore
        rotator = hp.Rotator(rot=[lat_angle, lon_angle], deg=True, inv=True)
        x = hp.reorder(inputs, n2r=True)
        x = rotator.rotate_map_alms(x, use_pixel_weights=False)
        return hp.reorder(x, r2n=True)

    x = tf.map_fn(rotate_map, inputs)
    x.set_shape(inputs.shape)  # type: ignore
    x = tf.transpose(x)
    return x, y


def to_tf(ds, core, batch_size=16, prerotate=True, for_unet=False):
    npix = hp.nside2npix(core.nside)
    dataset = tf.data.Dataset.from_generator(
        lambda: ds,
        output_signature=(
            tf.TensorSpec(shape=(core.ndups, core.npols, npix), dtype=tf.float32),  # type: ignore
            tf.TensorSpec(shape=(core.ndups,), dtype=tf.float32),  # type: ignore
        ),
    )

    # help fix a issue with TF not knowing the number of batches per epoch
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(ds)))

    # takes us from 1 element of (ndups, npols, npix) to ndup elements of (npols, npix)
    dataset = dataset.apply(tf.data.Dataset.unbatch)

    # dataset = dataset.map(lambda x, y: (tf.transpose(x), y))
    if prerotate:
        # apply rotations, do this before cache because its slow
        dataset = dataset.map(rotate_ds, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.cache()

    if not prerotate:
        # apply rotations, do this before cache because its slow
        dataset = dataset.map(rotate_ds, num_parallel_calls=tf.data.AUTOTUNE)

    if for_unet:
        dataset = dataset.map(lambda x, y: (x, x))

    dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    dataset = dataset.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


@keras.saving.register_keras_serializable()
class LinearWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup schedule."""

    def __init__(
        self,
        after_warmup_lr_sched: keras.optimizers.schedules.LearningRateSchedule | float,
        warmup_steps: int,
        warmup_learning_rate: float,
        name: str | None = None,
    ):
        """Add linear warmup schedule to a learning rate schedule.

        warmup_lr is the initial learning rate, the final learning rate of the
        init_warmup period is the initial learning rate of lr_schedule in use.
        The learning rate at each step linearly increased according to the following
        formula:
          learning_rate = warmup_lr + step / warmup_steps
                        * (final_warmup_lr - warmup_lr).
        Using warmup overrides the learning rate schedule by the number of warmup
        steps.

        Args:
          after_warmup_lr_sched: keras.optimizers.schedules .LearningRateSchedule
            or a constant.
          warmup_steps: Number of the warmup steps.
          warmup_learning_rate: Initial learning rate for the warmup.
          name: Optional, name of warmup schedule.
        """
        super().__init__()
        self._name = name
        self._after_warmup_lr_sched = after_warmup_lr_sched
        self._warmup_steps = warmup_steps
        self._init_warmup_lr = warmup_learning_rate
        if isinstance(
            after_warmup_lr_sched, keras.optimizers.schedules.LearningRateSchedule
        ):
            self._final_warmup_lr = after_warmup_lr_sched(warmup_steps)
        else:
            self._final_warmup_lr = tf.cast(after_warmup_lr_sched, dtype=tf.float32)

    def __call__(self, step: int):

        global_step = tf.cast(step, dtype=tf.float32)
        # print("Global step", global_step, flush=True)

        linear_warmup_lr = self._init_warmup_lr + global_step / self._warmup_steps * (
            self._final_warmup_lr - self._init_warmup_lr
        )

        if isinstance(
            self._after_warmup_lr_sched,
            keras.optimizers.schedules.LearningRateSchedule,
        ):
            after_warmup_lr = self._after_warmup_lr_sched(step)
        else:
            after_warmup_lr = tf.cast(self._after_warmup_lr_sched, dtype=tf.float32)

        lr = tf.cond(
            global_step < self._warmup_steps,
            lambda: linear_warmup_lr,
            lambda: after_warmup_lr,
        )
        return lr

    def get_config(self):
        if isinstance(
            self._after_warmup_lr_sched,
            keras.optimizers.schedules.LearningRateSchedule,
        ):
            config = {
                "after_warmup_lr_sched": self._after_warmup_lr_sched.get_config()
            }  # pytype: disable=attribute-error
        else:
            config = {
                "after_warmup_lr_sched": self._after_warmup_lr_sched
            }  # pytype: disable=attribute-error

        config.update(
            {
                "warmup_steps": self._warmup_steps,
                "warmup_learning_rate": self._init_warmup_lr,
                "name": self._name,
            }
        )
        return config


@keras.saving.register_keras_serializable()  # type: ignore
class SCNBlock(keras.Model):  # type: ignore

    def __init__(
        self,
        filters,
        n_mid=None,
        n_out=None,
        n_neighbors=8,
        cheb_degree=7,
        cheb_init=None,
        cheb_act=None,
        cheb_bias=True,
        cheb_batch=True,
        batch_size=None,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.n_neighbors = n_neighbors
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
        self.batch_size = batch_size
        self.activation = activation

        self.res_conv_1d = Conv1D(self.filters, kernel_size=1)
        self.cheb = HealpyChebyshev(
            K=self.cheb_degree,
            initializer=self.cheb_init,
            activation=self.cheb_act,
            use_bias=self.cheb_bias,
            use_bn=self.cheb_batch,
            depth_wise=True,
        )

    def build(self, input_shape):
        nside = hp.npix2nside(input_shape[1])
        self.res_conv_1d.build(input_shape)

        self.gcnn = HealpyGCNN(
            int(nside),
            np.arange(input_shape[1]),
            layers=[
                self.cheb,
                # LayerNormalization(),
                Conv1D(self.n_mid, kernel_size=1),
                LayerNormalization(),
                Activation(self.activation),
                Conv1D(self.n_out, kernel_size=1),
                # LayerNormalization(),
            ],
            verbose=False,
            max_batch_size=self.batch_size,
            initial_Fin=input_shape[-1],
        )
        self.gcnn.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        res = inputs
        if res.shape[-1] != self.filters:
            res = self.res_conv_1d(res)

        x = self.gcnn(inputs)
        return x + res

    def compute_output_shape(self, input_shape):
        output = (input_shape[0], input_shape[1], self.n_out)
        return output


@keras.saving.register_keras_serializable()
def SCNUNet(
    input_shape,
    base_channels=8,
    n_neighbors=8,
    blocks=[1, 1, 1, 1, 1, 1, 1],
    pools=[1, 1, 1, 1, 1, 1, 1],
    cheb_degree=7,
    cheb_init=None,
    cheb_act=None,
    cheb_bias=True,
    cheb_batch=True,
    max_batch_size=64,
    n_bottleneck=0,
    token_dim=256,
    channel_dim=2048,
    name="SCNUNet",
    scn_act="relu",
):
    inputs = keras.Input(shape=input_shape)
    channels = [base_channels * 2**p for p in range(len(blocks))]
    skips = []

    x = inputs
    for block in range(len(blocks)):
        for _ in range(blocks[block]):
            x = SCNBlock(
                channels[block],
                n_neighbors=n_neighbors,
                cheb_degree=cheb_degree,
                cheb_init=cheb_init,
                cheb_act=cheb_act,
                cheb_bias=cheb_bias,
                cheb_batch=cheb_batch,
                batch_size=max_batch_size,
                activation=scn_act,
            )(x)
        skips.append(x)
        x = HealpyPool(p=pools[block], pool_type="AVG")(x)

    # bottleneck
    for i in range(n_bottleneck):
        x = mixer_block(token_dim, channel_dim, name=f"Bottleneck_{i}")(x)
    if n_bottleneck > 0:
        x = layers.LayerNormalization(name="Bottleneck_norm")(x)

    for block, skip in zip(reversed(range(len(blocks))), reversed(skips)):
        x = HealpyPseudoConv_Transpose(pools[block], channels[block])(x)
        # x = Concatenate()([x, skip])

        for _ in range(blocks[block]):
            x = SCNBlock(
                channels[block],
                n_neighbors=n_neighbors,
                cheb_degree=cheb_degree,
                cheb_init=cheb_init,
                cheb_act=cheb_act,
                cheb_bias=cheb_bias,
                cheb_batch=cheb_batch,
                batch_size=max_batch_size,
            )(x)

    x = Conv1D(input_shape[-1], 1)(x)
    return keras.Model(inputs=inputs, outputs=x, name=name)  # type: ignore


@keras.saving.register_keras_serializable()  # type: ignore
class mlp_block(layers.Layer):
    def __init__(self, hidden_dim=512, activation="gelu", name="MLPBlock"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.activation = activation

    def build(self, input_shape):
        self.d1 = layers.Dense(self.hidden_dim, activation=self.activation)
        self.d2 = layers.Dense(input_shape[-1])

    def call(self, inputs):
        x = self.d1(inputs)
        return self.d2(x)

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "name": self.name,
        }


@keras.saving.register_keras_serializable()  # type: ignore
class mixer_block(layers.Layer):
    def __init__(
        self,
        token_dim=256,
        channel_dim=2048,
        name="MLPMixerBlock",
        activation="gelu",
    ):
        super().__init__(name=name)
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.activation = activation

        self.norm = layers.LayerNormalization()
        self.perm_1 = layers.Permute((2, 1))
        self.perm_2 = layers.Permute((2, 1))
        self.token_mixing = mlp_block(
            token_dim, activation=activation, name="TokenMixing"
        )
        self.channel_mixing = mlp_block(
            channel_dim, activation=activation, name="ChannelMixing"
        )

    def call(self, inputs):
        y = self.norm(inputs)
        y = self.perm_1(y)
        y = self.perm_2(self.token_mixing(self.perm_1(y)))
        y = self.perm_2(y)
        x = inputs + y
        y = self.norm(x)
        return x + self.channel_mixing(y)

    def get_config(self):
        return {
            "token_dim": self.token_dim,
            "channel_dim": self.channel_dim,
            "activation": self.activation,
            "name": self.name,
        }


def get_model(
    full_model,
    ff_layers=3,
    token_dim=256,
    channel_dim=2048,
    activation="relu",
    name="SCNReg",
):
    encoder_input = full_model.input
    encoder_output = None
    for layer in full_model.layers:
        if "healpy_pool_6" == layer.name:
            encoder_output = layer.output
            break
    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_output)  # type: ignore
    encoder_model.trainable = False

    x = encoder_model.output
    for i in range(ff_layers):
        x = mixer_block(
            token_dim, channel_dim, activation=activation, name=f"MLPMixerBlock_{i}"
        )(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = Dense(1, kernel_initializer="zeros")(x)

    return keras.Model(inputs=encoder_input, outputs=x, name=name)  # type: ignore


def plot_weights(model, save_dir):
    print("Plotting weights to directory", save_dir, flush=True)
    os.makedirs(save_dir, exist_ok=True)
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            filename = os.path.join(save_dir, f"{layer.name}.png")
            plt.figure()
            for i, weight in enumerate(weights):
                print(weight.shape, flush=True)
                # Plot each weight map
                for j in range(weight.shape[-1]):
                    test_map = weight[..., j]

                    try:
                        hp.mollview(test_map, nest=True)
                    except:
                        pass
            plt.savefig(filename)
            plt.close()


def unet_plots(model, ds, core, name):
    predictions = model.predict(ds, verbose=2)
    sim = list(ds.take(1).as_numpy_iterator())[0]
    sim = sim[0][0]
    sim = sim.flatten()
    test_map = predictions[0, :, 0].flatten()
    np.save(f"{core.dirs['model']}/{core.name}-unet-preds.npy", predictions)
    hp.mollview(test_map, title="Predictions", nest=True)
    os.makedirs(f"{core.dirs['plot']}/{core.name}", exist_ok=True)
    plt.savefig(f"{core.dirs['plot']}/{core.name}/unet-preds.png")
    plt.close()

    sim_cl = hp.anafast(hp.reorder(sim, n2r=True), lmax=core.lmax)
    cl = hp.anafast(hp.reorder(test_map, n2r=True), lmax=core.lmax)
    ells = np.arange(len(cl))
    plt.figure()
    plt.plot(ells, ells * (ells + 1) * cl, label="pred")  # type: ignore
    plt.plot(ells, ells * (ells + 1) * sim_cl, "--", label="sim")  # type: ignore
    plt.legend()
    plt.savefig(f"{core.dirs['plot']}/{core.name}/{name}-cl.png")
    plt.close()

    plt.figure()
    plt.semilogx(ells, ells * (ells + 1) * cl, label="pred")
    plt.semilogx(ells, ells * (ells + 1) * sim_cl, "--", label="sim")
    plt.legend()
    plt.savefig(f"{core.dirs['plot']}/{core.name}/{name}-cl-sx.png")
    plt.close()

    plt.figure()
    plt.semilogy(ells, ells * (ells + 1) * cl, label="pred")
    plt.semilogy(ells, ells * (ells + 1) * sim_cl, "--", label="sim")
    plt.legend()
    plt.savefig(f"{core.dirs['plot']}/{core.name}/{name}-cl-sy.png")
    plt.close()

    plt.figure()
    plt.loglog(ells, ells * (ells + 1) * cl, label="pred")
    plt.loglog(ells, ells * (ells + 1) * sim_cl, "--", label="sim")
    plt.legend()
    plt.savefig(f"{core.dirs['plot']}/{core.name}/{name}-cl-ll.png")
    plt.close()


def main():
    args = [
        "settings/n128.json",
        "--nsims",
        "1000",
        "--narray",
        "1",
        "--pols",
        "T",
    ]
    core = Core(args)
    fisher = get_fisher(core.file, "fisher_iso")
    npix = hp.nside2npix(core.nside)

    model_file = f"{core.dirs['model']}/{core.name}-{core.slurm.job}-unet-small.keras"
    reg_file = f"{core.dirs['model']}/{core.name}-{core.slurm.job}-small-reg.keras"
    strategy = tf.distribute.MirroredStrategy()
    metrics = []

    if not os.path.exists(model_file):
        BATCH_SIZE = 8

        f = 10
        ds = HDF5Dataset(core.file, x_name="alm", y_name="fnl")
        train_ds, test_ds, val_ds = ds.split(0.7 / f, 0.2 / f, 0.1 / f)
        decay_steps = len(train_ds) // BATCH_SIZE  # once per epoch
        train_tf, test_tf, val_tf = (
            to_tf(x, core, BATCH_SIZE, True, for_unet=True)
            for x in [train_ds, test_ds, val_ds]
        )

        with strategy.scope():
            learning_rate = ExponentialDecay(4e-3, decay_steps, 0.95, staircase=True)
            # learning_rate = LinearWarmup(learning_rate, decay_steps * 10, 1e-5)

            unet = SCNUNet((npix, 1))
            unet.compile(
                optimizer=AdamW(learning_rate),  # type: ignore
                loss="mse",
                metrics=metrics,  # type: ignore
            )

        unet.summary()

        tf_dir = f"{core.dirs['tb']}/{core.name}-unet"
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            TensorBoard(log_dir=tf_dir),
            ModelCheckpoint(tf_dir, monitor="val_loss", save_best_only=True),
        ]
        try_init_wandb(notes="model testing", tags=["SCNUNet"], append_to=callbacks)

        history = unet.fit(
            train_tf,
            epochs=30,
            validation_data=val_tf,
            callbacks=callbacks,
            verbose=2,
        )
        unet.evaluate(test_tf, verbose=2)  # type: ignore
        unet.save(model_file)

        # plot_weights(unet, f"{core.dirs['plot']}/{core.name}/unet-weights")
        unet_plots(unet, test_tf, core, f"{core.name}-{core.slurm.job}")
        del train_tf, test_tf, val_tf
    else:
        print("Loading model from file", model_file, flush=True)
        with strategy.scope():
            unet = keras.saving.load_model(  # type: ignore
                model_file,
                custom_objects={
                    "SCNBlock": SCNBlock,
                    "HealpyPool": HealpyPool,
                    "HealpyPseudoConv_Transpose": HealpyPseudoConv_Transpose,
                    "LinearWarmup": LinearWarmup,
                },
            )

    #####################################################################
    BATCH_SIZE = 64

    f = 1
    ds = HDF5Dataset(core.file, x_name="alm", y_name="fnl")
    train_ds, test_ds, val_ds = ds.split(0.9 / f, 0.05 / f, 0.05 / f)
    train_tf, test_tf, val_tf = (
        to_tf(x, core, BATCH_SIZE) for x in [train_ds, test_ds, val_ds]
    )
    decay_steps = len(train_ds) // BATCH_SIZE  # once per epoch

    with strategy.scope():
        learning_rate = ExponentialDecay(4e-3, decay_steps * 3, 0.95, staircase=False)
        learning_rate = LinearWarmup(learning_rate, decay_steps * 10, 1e-8)

        model = get_model(unet)
        model.build((BATCH_SIZE, npix, 1))
        model.compile(
            optimizer=AdamW(learning_rate, global_clipnorm=1),  # type: ignore
            loss="mse",
            metrics=[],  # type: ignore
        )

    model.summary()

    tf_dir = f"{core.dirs['tb']}/{core.name}-reg"
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        TensorBoard(log_dir=tf_dir),
        ModelCheckpoint(tf_dir, monitor="val_loss", save_best_only=True),
    ]
    try_init_wandb(notes="model testing", tags=["SCNUNet"], append_to=callbacks)

    history = model.fit(
        train_tf,
        epochs=300,
        validation_data=val_tf,
        callbacks=callbacks,
        verbose=2,
    )
    model.evaluate(test_tf, verbose=2)  # type: ignore
    model.save(reg_file)

    ###########################################################

    preds = model.predict(test_tf, verbose=2).flatten() * 100  # type: ignore
    truth = np.concatenate([y for _, y in test_tf]) * 100  # type: ignore

    name = f"SCNUReg-{core.slurm.job}"
    save_base = f"{core.dirs['plot']}/{core.name}"
    os.makedirs(save_base, exist_ok=True)

    print_errors(truth, preds, fisher)
    plot_metrics(history, metrics=["loss"], save_file=f"{save_base}/{name}-loss.png")
    plot_predictions(
        truth,
        preds,
        fisher=fisher,
        save_file=f"{save_base}/{name}-preds.png",
    )
    plot_histogram(truth, preds, save_file=f"{save_base}/{name}-hist.png")


if __name__ == "__main__":
    sys.exit(main())
