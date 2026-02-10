import os
import sys
import math
import logging
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from pixell import reproject, lensing, enmap, utils

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
    Healpy_ResidualLayer,
    HealpyPseudoConv_Transpose,
)

from mlpng import Core
from mlpng.utils import try_init_wandb, make_trainer_plots, setup_logging
from mlpng.utils.dataloaders import KappaDataset
from mlpng.utils.callbacks import RMSELoss, rmse_metrics

tf.get_logger().setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def get_fnl_model(input_shape, max_batch_size=32, n_out=1):
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


@tf.keras.saving.register_keras_serializable()
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nside,
        npix,
        fin,
        fout,
        activation,
        max_batch_size,
        K,
        use_bn=True,
        use_bias=False,
        pool=True,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.nside = nside
        self.npix = npix
        self.fin = fin
        self.fout = fout
        self.activation_fn = activation
        self.max_batch_size = max_batch_size
        self.K = K
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.pool = pool
        indices = np.arange(npix)

        enc_layers = [
            HealpyChebyshev(
                K=K,
                Fout=fout,
                activation=activation,
                use_bn=use_bn,
                use_bias=use_bias,
            ),
            HealpyChebyshev(
                K=K,
                Fout=fout,
                activation=activation,
                use_bn=use_bn,
                use_bias=use_bias,
            ),
        ]
        if dropout_rate > 0.0:
            enc_layers.append(Dropout(dropout_rate))

        self.body = HealpyGCNN(
            nside=nside,
            indices=indices,
            layers=enc_layers,
            n_neighbors=8,
            max_batch_size=max_batch_size,
            initial_Fin=fin,
        )

        self.pooler = None
        if pool:
            self.pooler = HealpyGCNN(
                nside=nside,
                indices=indices,
                layers=[HealpyPool(1, "AVG")],
                n_neighbors=8,
                max_batch_size=max_batch_size,
                initial_Fin=fin,
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nside": self.nside,
                "npix": self.npix,
                "fin": self.fin,
                "fout": self.fout,
                "activation": self.activation_fn,
                "max_batch_size": self.max_batch_size,
                "K": self.K,
                "use_bn": self.use_bn,
                "use_bias": self.use_bias,
                "pool": self.pool,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, inputs, training=False):
        x = skip = self.body(inputs, training=training)
        if self.pool:
            x = self.pooler(x, training=training)
        return x, skip


@tf.keras.saving.register_keras_serializable()
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nside,
        npix,
        fin,
        fout,
        activation,
        K,
        max_batch_size,
        upsample=True,
        use_bn=True,
        use_bias=False,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.nside = nside
        self.npix = npix
        self.fin = fin
        self.fout = fout
        self.activation_fn = activation
        self.max_batch_size = max_batch_size
        self.K = K
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.upsample = upsample

        indices = np.arange(npix)

        dec_layers = []
        if upsample:
            dec_layers.append(HealpyPseudoConv_Transpose(1, fout))
        dec_layers.extend(
            [
                HealpyChebyshev(
                    K=K,
                    Fout=fout,
                    activation=activation,
                    use_bn=use_bn,
                    use_bias=use_bias,
                ),
                HealpyChebyshev(
                    K=K,
                    Fout=fout,
                    activation=activation,
                    use_bn=use_bn,
                    use_bias=use_bias,
                ),
            ]
        )
        if dropout_rate > 0.0:
            dec_layers.append(Dropout(dropout_rate))

        self.body = HealpyGCNN(
            nside=nside,
            indices=indices,
            layers=dec_layers,
            n_neighbors=8,
            max_batch_size=max_batch_size,
            initial_Fin=fin,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nside": self.nside,
                "npix": self.npix,
                "fin": self.fin,
                "fout": self.fout,
                "activation": self.activation_fn,
                "K": self.K,
                "max_batch_size": self.max_batch_size,
                "upsample": self.upsample,
                "use_bn": self.use_bn,
                "use_bias": self.use_bias,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, inputs, skip, training=False):
        x = self.body(inputs, training=training)
        if skip is not None:
            x = x + skip
        return x


@tf.keras.saving.register_keras_serializable()
class ResidualHealpyUNet:
    def __init__(
        self,
        input_shape,
        activation="sigmoid",
        max_batch_size=32,
        dropout_rate=0.1,
        use_bn=True,
        use_bias=True,
    ):
        self.input_shape = input_shape
        self.activation = activation
        self.max_batch_size = max_batch_size
        self.npix = input_shape[1]
        self.nside = hp.npix2nside(self.npix)
        self.npol = input_shape[2]
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.use_bias = use_bias

    def get_config(self):
        return {
            "input_shape": self.input_shape,
            "activation": self.activation,
            "max_batch_size": self.max_batch_size,
            "dropout_rate": self.dropout_rate,
            "use_bn": self.use_bn,
            "use_bias": self.use_bias,
        }

    def get_model(self):
        depth = int(math.log2(self.nside))  # - 1

        base_channels = [self.npol] + [2 ** (i + 3) for i in range(depth)]
        level_npixels = [self.npix // (4**i) for i in range(depth + 1)]
        level_nsides = [hp.npix2nside(npix) for npix in level_npixels]
        Ks = [1 + (2 * (i // 3)) for i in range(depth + 1)]
        # Ks = [3 for _ in range(depth + 1)]
        logger.debug(f"Using Ks: {Ks}")

        inputs = tf.keras.Input(shape=self.input_shape[1:])
        x = inputs
        # x = layers.BatchNormalization()(x)

        skips = []
        for i in range(depth):
            block = EncoderBlock(
                level_nsides[i],
                level_npixels[i],
                fin=base_channels[i],
                fout=base_channels[i + 1],
                activation=self.activation,
                max_batch_size=self.max_batch_size,
                K=Ks[i],
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                use_bn=self.use_bn,
            )
            x, skip = block(x)
            skips.append(skip)

        bottleneck = HealpyGCNN(
            nside=level_nsides[depth],
            indices=np.arange(level_npixels[depth]),
            layers=[
                HealpyChebyshev(
                    K=Ks[-1],
                    Fout=base_channels[-1],
                    activation=self.activation,
                    use_bias=self.use_bias,
                    use_bn=self.use_bn,
                ),
                HealpyChebyshev(
                    K=Ks[-1],
                    Fout=base_channels[-1],
                    activation=self.activation,
                    use_bias=self.use_bias,
                    use_bn=self.use_bn,
                ),
            ],
            n_neighbors=8,
            max_batch_size=self.max_batch_size,
            initial_Fin=base_channels[-1],
            name="bottleneck",
        )
        x = bottleneck(x)

        for i in reversed(range(1, depth + 1)):
            block = DecoderBlock(
                level_nsides[i],
                level_npixels[i],
                fin=base_channels[i],
                fout=base_channels[i],
                activation=self.activation,
                max_batch_size=self.max_batch_size,
                K=Ks[i],
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                use_bn=self.use_bn,
            )
            x = block(x, skips[i - 1])

        # final block to account for upscaling to phi
        block = DecoderBlock(
            level_nsides[0],
            level_npixels[0],
            fin=base_channels[0],
            fout=base_channels[0],
            activation=None,
            max_batch_size=self.max_batch_size,
            K=Ks[0],
            dropout_rate=0.0,
            use_bias=self.use_bias,
            use_bn=self.use_bn,
        )
        x = block(x, None)

        # output
        output_head = HealpyGCNN(
            nside=self.nside,
            indices=np.arange(self.npix),
            layers=[
                HealpyChebyshev(
                    K=1,
                    Fout=base_channels[0],
                    activation=None,
                    use_bn=False,
                    use_bias=True,
                ),
            ],
            n_neighbors=8,
            max_batch_size=self.max_batch_size,
            initial_Fin=self.npol,
            name="output",
        )
        outputs = output_head(x)
        return tf.keras.Model(inputs, outputs)


def train_u_net(
    core,
    splits,
    duplicates,
    cache_file,
    batch_size,
    max_epochs,
    initial_LR,
    unet_keras_file,
    callbacks,
    strategy,
    shapes,
    plot_prefix,
    tf_cache,
    tf_mem_cache,
):
    # Determine cache strategy based on tf_cache and tf_mem_cache flags
    if tf_cache:
        # Use disk-based file cache
        cache_file_arg = cache_file
        logger.debug(f"Using disk-based cache: {cache_file_arg}")
    elif tf_mem_cache:
        # Use in-memory cache (empty string)
        cache_file_arg = ""
        logger.debug("Using in-memory cache")
    else:
        # Disable caching by passing None
        cache_file_arg = None
        logger.debug("Cache disabled")

    ds = KappaDataset.fromCore(
        core,
        x_output="lensed",
        y_output="kappa",
        kappa_scale=np.sqrt(1e7),
        phi_scale=1.0,
    )
    train, val, test = ds.split(
        train_size=splits[0],
        val_size=splits[1],
        test_size=splits[2],
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        cache_file=cache_file_arg,
        gen_batch_size=4,
        buffer_size=8,
    )
    logger.debug("Creating U-Net model and training data")

    # calculate the number of steps per epoch and decay steps for use later
    epoch_steps = math.ceil(core.total_sims * splits[0] * duplicates[0] // batch_size)
    decay_steps = epoch_steps * 3
    logger.debug(f"Steps per epoch: {epoch_steps}")

    with strategy.scope():
        learning_rate = ExponentialDecay(initial_LR, decay_steps, 0.96, staircase=True)

        u_net = ResidualHealpyUNet(
            (None, core.npix, core.npols),
            activation="gelu",
            max_batch_size=batch_size,
        ).get_model()

        logger.debug("Compiling")
        u_net.compile(
            optimizer=AdamW(learning_rate, weight_decay=4.5e-7, amsgrad=True),
            loss="mse",
            metrics=rmse_metrics(shapes) + ["mse", "mae"],
        )

    if tf_cache:
        logger.debug("Creating test cache")
        for _, y in test:
            pass
        logger.debug("Creating val cache")
        for _ in val:
            pass
        logger.debug("Creating train cache")
        for _ in train:
            pass
        logger.debug("Caches created")

    u_net.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )

    u_net.save(unet_keras_file)

    test_kappa = np.concatenate([y for _, y in test])
    pred_kappa = u_net.predict(test, verbose=0)

    # Convert kappa to phi
    pred_phi = ds.kappa_to_phi(pred_kappa)
    test_phi = ds.kappa_to_phi(test_kappa)

    # pick the first item, and remove batch dim if present
    print(f"test_phi shape: {test_phi.shape}, pred_phi shape: {pred_phi.shape}")
    test_phi = test_phi[0]
    pred_phi = pred_phi[0]

    test_phi_ring = hp.reorder(test_phi, n2r=True)
    pred_phi_ring = hp.reorder(pred_phi, n2r=True)

    for map, name in [(test_phi_ring, "True"), (pred_phi_ring, "Predicted")]:
        hp.mollview(map, title=f"{name} phi map", unit="phi")
        plt.savefig(f"{plot_prefix}-{name.lower()}-phi-map.png")

    hp.mollview(test_phi - pred_phi, title="Residual (Nest)", cmap="RdBu_r")
    plt.savefig(f"{plot_prefix}-residual-nest-phi-map.png")

    hp.mollview(test_phi_ring - pred_phi_ring, title="Residual (Ring)", cmap="RdBu_r")
    plt.savefig(f"{plot_prefix}-residual-ring-phi-map.png")

    plt.figure()
    for map, name in [(test_phi_ring, "True"), (pred_phi_ring, "Predicted")]:
        cl = hp.anafast(
            hp.remove_dipole(map), lmax=core.lmax, pol=False, use_pixel_weights=True
        )
        plt.loglog(cl, label=name)
    plt.legend()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^{\phi\phi}$")
    plt.savefig(f"{plot_prefix}-phi-cl-ring.png")

    plt.figure()
    for map, name in [(test_phi, "True (nest)"), (pred_phi, "Predicted (nest)")]:
        cl = hp.anafast(
            hp.remove_dipole(map), lmax=core.lmax, pol=False, use_pixel_weights=True
        )
        plt.loglog(cl, label=name)
    plt.legend()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^{\phi\phi}$")
    plt.savefig(f"{plot_prefix}-phi-cl-nest.png")

    return u_net


def train_fnl(
    core,
    split,
    duplicates,
    cache_file,
    batch_size,
    max_epochs,
    initial_LR,
    save_file,
    callbacks,
    strategy,
    shapes,
    plot_prefix,
    tf_cache,
    tf_mem_cache,
):
    # Determine cache strategy based on tf_cache and tf_mem_cache flags
    if tf_cache:
        # Use disk-based file cache
        cache_file_arg = cache_file
        logger.debug(f"Using disk-based cache: {cache_file_arg}")
    elif tf_mem_cache:
        # Use in-memory cache (empty string)
        cache_file_arg = ""
        logger.debug("Using in-memory cache")
    else:
        # Disable caching by passing None
        cache_file_arg = None
        logger.debug("Cache disabled")

    ds = KappaDataset.fromCore(
        core,
        x_output="unlensed",
        y_output="fnl",
        lensed=False,
    )
    train, val, test = ds.split(
        train_size=split[0],
        val_size=split[1],
        test_size=split[2],
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        cache_file=cache_file_arg,
        gen_batch_size=core.slurm.n_cpus,
    )
    if tf_cache:
        logger.debug("Creating fnl caches")
        for _ in train:
            pass
        logger.debug("Fnl train cache created")
        for _ in val:
            pass
        logger.debug("Fnl val cache created")
        for _ in test:
            pass
        logger.debug("Fnl caches created")

    # calculate the number of steps per epoch and decay steps for use later
    epoch_steps = math.ceil(core.total_sims * split[0] * duplicates[0] // batch_size)
    decay_steps = epoch_steps * 3

    with strategy.scope():
        learning_rate = ExponentialDecay(initial_LR, decay_steps, 0.95, staircase=True)

        fnl_model = get_fnl_model(
            (None, core.npix, core.npols), batch_size, len(shapes)
        )

        fnl_model.compile(
            optimizer=AdamW(learning_rate),
            loss=RMSELoss(),
            metrics=rmse_metrics(shapes),
        )

    fnl_model.summary()

    # and finally we fit the model
    fnl_model.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )

    logger.info("Saving Fnl model to %s", save_file)
    fnl_model.save(save_file)

    truth = np.concatenate([y for _, y in test])
    preds = np.concatenate([fnl_model.predict(b, verbose=0) for b, _ in test])

    vals = truth.ravel()
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=50, color="C0", alpha=0.8)
    plt.xlabel("fnl (truth)")
    plt.ylabel("Count")
    plt.title(f"Distribution of truth values (N={vals.size})")
    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}-fnl-truth-distribution.png",
    )

    truth = np.array(truth).reshape(-1, 1)

    # Calculate metrics
    fnl_error = preds - truth
    logger.info(f"Mean absolute error: {np.mean(np.abs(fnl_error)):.4f}")
    logger.info(f"RMSE: {np.sqrt(np.mean(fnl_error**2)):.4f}")

    line = np.array([np.nanmin(truth), np.nanmax(truth)])
    sigma = core.get_likelihoods(False)[0]

    # Plot results
    plt.figure(figsize=(16, 4))

    plt.subplot(131)
    plt.scatter(truth, preds, alpha=0.5)
    plt.plot(line, line, "r--")
    plt.plot(line, line + sigma, "g--")
    plt.plot(line, line - sigma, "g--")
    plt.xlabel("True fnl")
    plt.ylabel("Predicted fnl")
    plt.title("fnl Prediction from unlensed Maps")

    plt.subplot(132)
    plt.hist(fnl_error, bins=50)
    plt.axvline(sigma, color="g", linestyle="--", label=f"+σ={sigma:.2f}")
    plt.axvline(-sigma, color="g", linestyle="--", label=f"-σ={-sigma:.2f}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("fnl Error Distribution")

    plt.subplot(133)
    plt.plot(preds, alpha=0.7, label="Predicted")
    plt.plot(truth, alpha=0.7, label="True")
    plt.xlabel("Sample Index")
    plt.ylabel("fnl Value")
    plt.title("fnl Values Over Test Set")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}-fnl-prediction-results.png",
    )
    return fnl_model


def run():
    core = Core()

    # start with some hard coded settings because
    batch_size = 8
    max_epochs = 300
    initial_LR = 1e-3

    data_fraction = 1.0
    unet_split = np.array([0.8, 0.1, 0.1]) * data_fraction
    unet_duplicates = [25, 10, 2]

    # fnl values come from the
    fnl_split = np.array([0.4, 0.1, 0.5]) * data_fraction
    fnl_duplicates = [25, 10, 2]

    final_split = np.array([0.8, 0.1, 0.1])  # * data_fraction
    final_duplicates = [1, 1, 1]

    strategy = tf.distribute.MirroredStrategy()  # use mirrored strategy for multi-GPU

    date_time = tf.timestamp().numpy().astype(int)
    # freeze to one run to reuse the cache
    # date_time = "1762263789"  # nside 256, no rotations
    run_name = f"trainer-{date_time}"
    logger.info(f"Run name: {run_name}")

    save_dir = f"{core.dirs['model']}/{core.name}"
    run_info = f"{core.shapes_str()}-{run_name}"
    plot_prefix = f"{core.dirs['plot']}/{core.name}/train/{core.slurm.job}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{core.dirs['plot']}/{core.name}", exist_ok=True)
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)

    # setups where we will
    unet_keras_file = f"{save_dir}/unet-{run_info}.keras"
    fnl_keras_file = f"{save_dir}/fnl-{run_info}.keras"

    temp_folder = os.environ.get("SCRATCH", "/tmp")
    cache_dir = f"{temp_folder}/tf_cache/{core.name}"
    os.makedirs(cache_dir, exist_ok=True)

    unet_cache = f"{cache_dir}/unet-{core.name}"
    fnl_cache = f"{cache_dir}/fnl-{core.name}"
    for file in [unet_keras_file, fnl_keras_file]:
        logger.debug(f"file {file} exists: {os.path.exists(file)}")

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True),
        tf.keras.callbacks.BackupAndRestore(
            f"{core.dirs['model']}/chkpts/{run_info}/",
            save_freq="epoch",
            delete_checkpoint=True,
        ),
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
            dir=core.dirs["wandb"],
            append_to=callbacks,
            patch_tb=core.use_tb,
            patch_logdir=tb_dir,
        )

    if not os.path.exists(unet_keras_file):
        u_net = train_u_net(
            core,
            unet_split,
            unet_duplicates,
            unet_cache,
            batch_size,
            max_epochs,
            initial_LR,
            unet_keras_file,
            callbacks,
            strategy,
            core.shapes,
            plot_prefix,
            core.tf_cache,
            core.tf_mem_cache,
        )
    else:
        logger.info("Loading U-Net from file: '%s'", unet_keras_file)
        u_net = tf.keras.models.load_model(unet_keras_file)

    if not os.path.exists(fnl_keras_file):
        fnl_model = train_fnl(
            core,
            fnl_split,
            fnl_duplicates,
            fnl_cache,
            batch_size,
            max_epochs,
            initial_LR,
            fnl_keras_file,
            callbacks,
            strategy,
            core.shapes,
            plot_prefix,
            core.tf_cache,
            core.tf_mem_cache,
        )
    else:
        logger.info("Loading model from %s", fnl_keras_file)
        fnl_model = tf.keras.models.load_model(fnl_keras_file)

    ds_3 = KappaDataset.fromCore(
        core, x_output="lensed", y_output="fnl", kappa_scale=np.sqrt(1e7)
    )
    _, _, test_3 = ds_3.split(
        train_size=final_split[0],
        val_size=final_split[1],
        test_size=final_split[2],
        to_tf=True,
        batch_size=batch_size,
        duplicates=final_duplicates,
        gen_batch_size=core.slurm.n_cpus,
    )

    # logger.info("Creating test_3 cache")
    # fnl_truth = np.concatenate([y for _, y in test_3])
    # logger.info("Done")

    res_arcmin = hp.nside2resol(core.nside, arcmin=True)
    res_rad = res_arcmin * utils.arcmin
    # snap to nearest resolution that evenly divides the sky (vertical)
    ny = int(round(np.pi / res_rad))
    res_fixed = np.pi / ny  # exact divisor for π
    logger.debug(
        f"healpy res (arcmin) = {res_arcmin:.6f}, using fixed res (arcmin) = {res_fixed / utils.arcmin:.6f}"
    )

    # build geometry with the fixed resolution
    shape, wcs = enmap.fullsky_geometry(res_fixed)

    def _delens_py(lensed, phi, nside=core.nside):
        delensed_maps = np.zeros_like(lensed)
        for i, (l_map, phi_map) in enumerate(zip(lensed, phi)):
            l_map = np.asarray(l_map)
            phi_map = np.asarray(phi_map)

            l_map = hp.reorder(l_map.flatten(), n2r=True).astype(np.float32)
            phi_map = hp.reorder(phi_map.flatten(), n2r=True).astype(np.float32)

            lensed_enmap = reproject.healpix2map(l_map, shape, wcs)
            phi_enmap = reproject.healpix2map(phi_map, shape, wcs)
            delensed_enmap = lensing.delens_map(lensed_enmap, phi_enmap)

            # Convert back to healpix and flatten
            m = reproject.map2healpix(delensed_enmap, nside)
            m = hp.remove_dipole(m, copy=False)
            delensed_maps[i] = m[:, None]

        return delensed_maps

    def _map_fn(pair, phi):
        lensed, fnl = pair
        delensed = tf.py_function(
            func=_delens_py,
            inp=[lensed, phi],
            Tout=tf.float32,
        )

        delensed.set_shape([None, core.npix, core.npols])
        fnl.set_shape([None, len(core.shapes)])
        return delensed, fnl

    kappa_preds = u_net.predict(test_3, verbose=0)
    phi_preds = ds_3.kappa_to_phi(kappa_preds)

    phi_ds = tf.data.Dataset.from_tensor_slices(phi_preds).batch(batch_size)
    delensed_ds = (
        tf.data.Dataset.zip((test_3, phi_ds))
        .map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    for map, _ in delensed_ds.take(1):
        m = map[0, :, 0].numpy()
        hp.mollview(m, title="Delensed map", unit="T")
        plt.savefig(f"{plot_prefix}-final-delensed-map.png")

        plt.figure()
        cl = hp.anafast(m, lmax=core.lmax, pol=False, use_pixel_weights=True)
        plt.loglog(cl, label="Delensed")
        plt.legend()
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
        plt.savefig(f"{plot_prefix}-final-delensed-cl.png")

    fnl_preds = fnl_model.predict(delensed_ds, verbose=2)
    fnl_truth = np.concatenate([y for _, y in delensed_ds])

    fnl_truth = fnl_truth.ravel()
    fnl_preds = fnl_preds.ravel()

    # Calculate metrics
    fnl_error = fnl_preds - fnl_truth
    logger.info(f"Mean absolute error: {np.mean(np.abs(fnl_error)):.4f}")
    logger.info(f"RMSE: {np.sqrt(np.mean(fnl_error**2)):.4f}")

    line = np.array([np.nanmin(fnl_truth), np.nanmax(fnl_truth)])
    sigma = core.get_likelihoods(True)[0]
    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(fnl_truth, fnl_preds, alpha=0.5)
    plt.plot(line, line, "r--")
    plt.plot(line, line + sigma, "g--")
    plt.plot(line, line - sigma, "g--")
    plt.xlabel("True fnl")
    plt.ylabel("Predicted fnl")
    plt.title("fnl Prediction from Delensed Maps")

    plt.subplot(132)
    plt.hist(fnl_error, bins=50)
    plt.axvline(sigma, color="g", linestyle="--", label=f"+σ={sigma:.2f}")
    plt.axvline(-sigma, color="g", linestyle="--", label=f"-σ={-sigma:.2f}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("fnl Error Distribution")

    plt.subplot(133)
    plt.plot(fnl_preds, alpha=0.7, label="Predicted")
    plt.plot(fnl_truth, alpha=0.7, label="True")
    plt.xlabel("Sample Index")
    plt.ylabel("fnl Value")
    plt.title("fnl Values Over Test Set")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}-final-fnl-prediction-delensed-results.png",
    )

    plt.figure(figsize=(6, 4))
    plt.hist(fnl_truth.ravel(), bins=50, color="C0", alpha=0.8)
    plt.xlabel("fnl (truth)")
    plt.ylabel("Count")
    plt.title(f"Distribution of truth values")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}-final-fnl-truth-distribution.png")


if __name__ == "__main__":
    setup_logging(__name__, level=logging.DEBUG)

    logger.info("Conda environment: %s", os.environ["CONDA_DEFAULT_ENV"])
    logger.info("Python executable: %s", sys.executable)
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("CUDA version: %s", tf.sysconfig.get_build_info()["cuda_version"])
    logger.info("cuDNN version: %s", tf.sysconfig.get_build_info()["cudnn_version"])

    sys.exit(run())
