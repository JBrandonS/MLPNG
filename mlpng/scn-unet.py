import os
import sys
import math
import logging

import healpy as hp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv1D,
    Dense,
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
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError  # type: ignore

import wandb

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
    HealpyPseudoConv_Transpose,
)


logger = setup_logging(__name__, level=logging.DEBUG)

# @keras.saving.register_keras_serializable()
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


# @tf.keras.saving.register_keras_serializable()  # type: ignore
class SCNBlock(keras.Model):  # type: ignore

    def __init__(
        self,
        filters,
        n_mid=None,
        n_mid_scale=4,
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
            n_mid = filters * n_mid_scale
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
        npix = input_shape[-2]
        nside = hp.npix2nside(npix)
        indices = np.arange(npix)

        self.res_conv_1d.build(input_shape)
        self.gcnn = HealpyGCNN(
            int(nside),
            indices,
            layers=[
                self.cheb,
                Conv1D(self.n_mid, kernel_size=1),
                LayerNormalization(),
                Activation(self.activation),
                Conv1D(self.n_out, kernel_size=1),
            ],
            verbose=False,
            max_batch_size=self.batch_size,
            initial_Fin=input_shape[-1],
        )
        self.gcnn.build(input_shape)

    def call(self, inputs, training=None, mask=None):
        res = inputs
        if res.shape[-1] != self.filters:
            res = self.res_conv_1d(res)

        x = self.gcnn(inputs)
        return x + res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_out)


# @tf.keras.saving.register_keras_serializable()
def SCNUNet(
    input_shape,
    base_channels=2,
    n_neighbors=8,
    cheb_degree=3,
    cheb_init=None,
    cheb_act=None,
    cheb_bias=True,
    cheb_batch=True,
    max_batch_size=64,
    n_bottleneck=1,
    token_dim=32,
    channel_dim=16,
    name="SCNUNet",
    scn_act="relu",
):
    nside = hp.npix2nside(input_shape[-2])
    n_layers = math.floor(math.log(nside, 2))
    channels = [base_channels * 2**p for p in range(n_layers)]
    inputs = keras.Input(shape=input_shape)

    skips = []
    x = inputs
    for layer in range(n_layers):
        x = SCNBlock(
            channels[layer],
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
        x = HealpyPool(p=1, pool_type="AVG")(x)

    # bottleneck
    for i in range(n_bottleneck):
        x = mixer_block(token_dim, channel_dim, name=f"Bottleneck_{i}")(x)
    if n_bottleneck > 0:
        x = layers.LayerNormalization(name="Bottleneck_norm")(x)

    # decoder
    for block, skip in zip(reversed(range(n_layers)), reversed(skips)):
        x = HealpyPseudoConv_Transpose(1, channels[block])(x)
        x = Concatenate()([x, skip])
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


# @tf.keras.saving.register_keras_serializable()  # type: ignore
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


# @tf.keras.saving.register_keras_serializable()  # type: ignore
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
        y = self.token_mixing(y)
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


def SCNEstimator(
    full_model,
    ff_layers=1,
    token_dim=256,
    channel_dim=128,
    mixer_activation="gelu",
    name="SCNEstimator",
):
    encoder_input = full_model.input
    encoder_output = None
    for layer in full_model.layers:
        if "Bottleneck_0" == layer.name:
            encoder_output = layer.output
            break

    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_output)  # type: ignore
    encoder_model.trainable = False

    x = encoder_model.output
    for i in range(ff_layers):
        x = mixer_block(
            token_dim,
            channel_dim,
            activation=mixer_activation,
            name=f"MLPMixerBlock_{i}",
        )(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = Dense(1, kernel_initializer="zeros")(x)
    x = Dense(1)(x)

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
    core = Core()
    fisher = get_fisher(core.file, "fisher_iso")
    npix = hp.nside2npix(core.nside)

    unet_file = f"{core.dirs['model']}/scnunet-{core.name}.keras"
    model_file = f"{core.dirs['model']}/scnestimator{core.name}.keras"

    unet_epochs = 40
    unet_batch_size = 64
    unet_decay_steps = int(core.total_sims * 0.8 * core.data_fraction / unet_batch_size)

    estimator_epochs = 40
    estimator_batch_size = 64
    estimator_decay_steps = int(
        core.total_sims * 0.8 * core.data_fraction / estimator_batch_size
    )

    metrics = [RootMeanSquaredError()]
    strategy = tf.distribute.MirroredStrategy()

    if not os.path.exists(unet_file):
        train, val, test = tfds_from_hdf5(
            core,
            "/map",
            "/map",
            data_fraction=core.data_fraction,
            batch_size=unet_batch_size,
            reshape_y=False,
            transpose_y=True,
            buffer=128,
            cache_prefix="unet",
        )

        with strategy.scope():
            learning_rate = ExponentialDecay(
                4e-3, unet_decay_steps, 0.95, staircase=True
            )
            learning_rate = LinearWarmup(learning_rate, unet_decay_steps, 1e-5)

            unet = SCNUNet((npix, core.npols))
            unet.compile(
                optimizer=AdamW(learning_rate, weight_decay=0.01, global_clipnorm=1.0),  # type: ignore
                loss="mse",
                metrics=metrics,
            )

        unet.summary()

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ]
        if core.use_tb:
            callbacks += [
                TensorBoard(log_dir=core.dirs["tb"]),
                ModelCheckpoint(
                    core.dirs["tb"], monitor="val_loss", save_best_only=True
                ),
            ]
        if core.use_wandb:
            try_init_wandb(
                notes="model testing",
                tags=["SCNUNet"],
                append_to=callbacks,
                patch_tb=core.use_tb,
            )

        history = unet.fit(
            train,
            epochs=unet_epochs,
            validation_data=val,
            callbacks=callbacks,
            verbose=1,
        )
        unet.evaluate(test, verbose=2)  # type: ignore
        unet.save(unet_file)

        if core.use_wandb:
            wandb.finish()

        unet_plots(unet, test, core, f"{core.name}-{core.slurm.job}")
        plot_metrics(
            history, metrics=["loss"], save_file=core.get_plot_file("unet_loss")
        )
    else:
        logger.info("Loading model from file: %s", model_file)
        with strategy.scope():
            unet = keras.saving.load_model(  # type: ignore
                unet_file,
                custom_objects={
                    "SCNBlock": SCNBlock,
                    "HealpyPool": HealpyPool,
                    "HealpyPseudoConv_Transpose": HealpyPseudoConv_Transpose,
                    "LinearWarmup": LinearWarmup,
                },
            )

    #####################################################################
    # Train estimator

    train, val, test = tfds_from_hdf5(
        core,
        "/map",
        "/fnl",
        data_fraction=core.data_fraction,
        batch_size=estimator_batch_size,
    )

    with strategy.scope():
        learning_rate = ExponentialDecay(
            4e-3, estimator_decay_steps, 0.95, staircase=False
        )
        learning_rate = LinearWarmup(learning_rate, estimator_decay_steps, 1e-8)

        estimator = SCNEstimator(unet)
        estimator.build((npix, core.npols))
        estimator.compile(
            optimizer=AdamW(learning_rate, weight_decay=0.01, global_clipnorm=1.0),  # type: ignore
            loss="mse",
            metrics=metrics,
        )

    estimator.summary()

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
            tags=["SCNUNet-Estimator"],
            append_to=callbacks,
            patch_tb=core.use_tb,
        )

    history = estimator.fit(
        train,
        epochs=estimator_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )
    estimator.evaluate(test, verbose=2)  # type: ignore
    estimator.save(model_file)

    preds = estimator.predict(test, verbose=2).flatten()
    truth = np.concatenate([y for _, y in test])

    print_errors(truth, preds, fisher)
    plot_metrics(history, metrics=["loss"], save_file=core.get_plot_file("est_loss"))
    plot_predictions(
        truth,
        preds,
        fisher=fisher,
        save_file=core.get_plot_file("est_predictions"),
    )
    plot_histogram(truth, preds, save_file=core.get_plot_file("est_histogram"))


if __name__ == "__main__":
    sys.exit(main())
