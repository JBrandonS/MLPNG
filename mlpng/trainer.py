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
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Add
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
    Healpy_ResidualLayer,
    HealpyPseudoConv_Transpose,
)

from mlpng import Core
from mlpng.utils import setup_logging, try_init_wandb, make_trainer_plots
from mlpng.utils.dataloaders import MapDataset, UnlensMapDataset, PhiMapDataset
from mlpng.utils.callbacks import RMSELoss, RMSELoss2, rmse_metrics, PowerSpectrumLoss

tf.get_logger().setLevel(logging.ERROR)
logger = setup_logging("mlpng.trainer", level=logging.DEBUG)


def get_u_net_model(
    input_shape, max_batch_size=32, activation=LeakyReLU(0.3), use_double=False
):
    nside = hp.npix2nside(input_shape[1])
    n_layers = math.floor(math.log(nside, 2))

    layers = []

    # Encoder path: progressively downsample and increase features
    encoder_channels = [2 ** (i + 5) for i in range(n_layers - 2)]
    n_encoder_layers = min(n_layers - 2, len(encoder_channels))

    for i in range(n_encoder_layers):
        fout = encoder_channels[i]

        # Convolutional blocks
        layers.append(
            HealpyChebyshev(
                K=3,
                Fout=fout,
                use_bn=True,
                use_bias=True,
                activation=activation,
            )
        )
        if use_double:
            layers.append(
                HealpyChebyshev(
                    K=3,
                    Fout=fout,
                    use_bn=True,
                    use_bias=True,
                    activation=activation,
                )
            )
        layers.append(Dropout(0.1))

        # Downsample
        layers.append(HealpyPool(1, "AVG"))

    # Bottleneck
    bottleneck_channels = encoder_channels[-1]
    layers.append(
        HealpyChebyshev(
            K=5,  # Larger receptive field at bottleneck
            Fout=bottleneck_channels,
            use_bn=True,
            use_bias=True,
            activation=activation,
        )
    )
    if use_double:
        layers.append(
            HealpyChebyshev(
                K=5,
                Fout=bottleneck_channels,
                use_bn=True,
                use_bias=True,
                activation=activation,
            )
        )
    layers.append(Dropout(0.1))

    # Decoder path: upsample and decrease features
    decoder_channels = encoder_channels[::-1][1:]  # Reverse and skip first

    for i, fout in enumerate(decoder_channels):
        # Upsample
        layers.append(HealpyPseudoConv_Transpose(1, fout))

        # Convolutional blocks
        layers.append(
            HealpyChebyshev(
                K=3,
                Fout=fout,
                use_bn=True,
                use_bias=True,
                activation=activation,
            )
        )
        if use_double:
            layers.append(
                HealpyChebyshev(
                    K=3,
                    Fout=fout,
                    use_bn=True,
                    use_bias=True,
                    activation=activation,
                )
            )
        layers.append(Dropout(0.1))

    # Final upsampling to original resolution
    layers.append(HealpyPseudoConv_Transpose(1, input_shape[-1]))

    # Final output layer - map to original number of channels
    layers.append(
        HealpyChebyshev(
            K=3,
            Fout=input_shape[-1],  # Same as input channels
            use_bn=False,
            # use_bias=True,
            activation=None,  # Linear output for reconstruction
        )
    )

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
):
    ds = PhiMapDataset.fromCore(core, lensed=True)
    train, val, test = ds.split(
        train_size=splits[0],
        val_size=splits[1],
        test_size=splits[2],
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        cache_file=cache_file,
        gen_batch_size=4,
    )
    logger.debug("Creating U-Net model and training data")

    # calculate the number of steps per epoch and decay steps for use later
    epoch_steps = math.ceil(core.total_sims * splits[0] * duplicates[0] // batch_size)
    decay_steps = epoch_steps * 3
    logger.debug(f"Steps per epoch: {epoch_steps}")

    with strategy.scope():
        learning_rate = ExponentialDecay(initial_LR, decay_steps, 0.96, staircase=True)

        u_net = get_u_net_model((None, core.npix, core.npols), core.nside)

        u_net.compile(
            optimizer=AdamW(
                learning_rate, weight_decay=5e-6, amsgrad=True, use_ema=True
            ),
            loss=PowerSpectrumLoss(core.lmax),
            metrics=rmse_metrics(shapes) + ["mse", "mae"],
        )

    logger.debug("Creating test cache")
    y_test = np.concatenate([y for _, y in test])
    logger.debug("Creating val cache")
    for _ in val:
        pass
    logger.debug("Creating train cache")
    for _ in train:
        pass
    logger.debug("Caches created")

    u_net_history = u_net.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )

    u_net.save(unet_keras_file)

    y_test = y_test.reshape(-1, *y_test.shape[-2:])
    pred = u_net.predict(test)

    for map, name in [(y_test[0, :, 0], "True"), (pred[0, :, 0], "Predicted")]:
        r_map = hp.reorder(map, n2r=True)
        plt.figure()
        hp.mollview(r_map, title=f"{name} phi map", unit="phi", cmap="viridis")
        plt.savefig(f"{plot_prefix}-{name}-phi-map.png")

    ells = np.arange(core.lmax + 1)
    ell_scale = ells * (ells + 1) / (2 * np.pi)
    plt.figure()
    for map, name in [(y_test[0, :, 0], "True"), (pred[0, :, 0], "Predicted")]:
        r_map = hp.reorder(map, n2r=True)
        cl = hp.anafast(r_map, lmax=core.lmax, pol=False, use_pixel_weights=True)
        plt.loglog(cl * ell_scale, label=name)

    plt.legend()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$(\ell * (\ell + 1) / (2 \pi) C_\ell^{\phi\phi})$")
    plt.savefig(f"{plot_prefix}-phi-cl.png")
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
):
    ds = MapDataset.fromCore(core, lensed=False)
    train, val, test = ds.split(
        train_size=split[0],
        val_size=split[1],
        test_size=split[2],
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        cache_file=cache_file,
        gen_batch_size=core.slurm.n_cpus * 4,
    )

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
    fnl_history = fnl_model.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=2,
    )

    logger.info("Saving Fnl model to %s", save_file)
    fnl_model.save(save_file)

    truth = np.concatenate([y for _, y in test])
    preds = fnl_model.predict(test, verbose=0)

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
    batch_size = 64
    max_epochs = 500
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
    run_name = f"notebook-{date_time}"
    logger.info(f"Run name: {run_name}")

    save_dir = f"{core.dirs['model']}/{core.name}"
    run_info = f"{core.shapes_str()}-{run_name}"
    plot_prefix = f"{core.dirs['plot']}/{core.name}/train/{core.slurm.job}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{core.dirs['plot']}/{core.name}", exist_ok=True)
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)

    # setups where we will
    unet_keras_file = f"{save_dir}/unet-{run_info}-2.keras"
    fnl_keras_file = f"{save_dir}/fnl-{run_info}-2.keras"
    final_keras_file = f"{save_dir}/final-{run_info}-2.keras"
    full_keras_file = f"{save_dir}/full-{run_info}-2.keras"

    unet_cache = f"{core.name}/unet-{core.name}"
    fnl_cache = f"{core.name}/fnl-{core.name}"
    final_cache = f"{core.name}/final-{core.name}"

    for file in [
        unet_keras_file,
        fnl_keras_file,
        final_keras_file,
        full_keras_file,
    ]:
        logger.debug(f"file {file} exists: {os.path.exists(file)}")

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
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
        )
    else:
        logger.info("Loading model from %s", fnl_keras_file)
        fnl_model = tf.keras.models.load_model(fnl_keras_file)

    ds_3 = MapDataset.fromCore(core, lensed=True)
    train_3, val_3, test_3 = ds_3.split(
        train_size=final_split[0],
        val_size=final_split[1],
        test_size=final_split[2],
        to_tf=True,
        batch_size=batch_size,
        duplicates=final_duplicates,
        cache_file=final_cache,
        gen_batch_size=core.slurm.n_cpus,
    )

    logger.info("Creating test_3 cache")
    fnl_truth = np.concatenate([y for _, y in test_3])
    logger.info("Done")

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
        # TODO test joblib parallel here
        for i, (l_map, phi_map) in enumerate(zip(lensed, phi)):
            # TODO add pol support here
            l_map = hp.reorder(l_map[:, 0], n2r=True).astype(np.float32)
            phi_map = hp.reorder(phi_map[:, 0], n2r=True).astype(np.float32)

            lensed_enmap = reproject.healpix2map(l_map, shape, wcs)
            phi_enmap = reproject.healpix2map(phi_map, shape, wcs)
            delensed_enmap = lensing.delens_map(lensed_enmap, phi_enmap)

            dm = reproject.map2healpix(delensed_enmap, nside)
            delensed_maps[i] = hp.reorder(dm, r2n=True)

        return delensed_maps

    def _map_fn(pair, phi):
        lensed, truth = pair
        delensed = tf.py_function(
            func=_delens_py,
            inp=[lensed, phi],
            Tout=tf.float32,
        )

        delensed.set_shape([None, core.npix, core.npols])
        truth.set_shape([None, 1])
        return delensed, truth

    phi_preds = u_net.predict(test_3, verbose=0)
    # phi_preds = np.array([hp.reorder(x, r2n=True), for x in phi_preds])

    phi_ds = tf.data.Dataset.from_tensor_slices(phi_preds).batch(batch_size)
    delensed_ds = (
        tf.data.Dataset.zip((test_3, phi_ds))
        .map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    for map, _ in delensed_ds.take(1):
        plt.figure()
        hp.mollview(
            hp.reorder(map[0, :, 0].numpy(), n2r=True),
            title="Delensed map",
            unit="T",
            cmap="viridis",
        )
        plt.savefig(f"{plot_prefix}-delensed-map.png")

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
        f"{plot_prefix}-fnl-prediction-delensed-results.png",
    )

    plt.figure(figsize=(6, 4))
    plt.hist(fnl_truth.ravel(), bins=50, color="C0", alpha=0.8)
    plt.xlabel("fnl (truth)")
    plt.ylabel("Count")
    plt.title(f"Distribution of truth values")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}-fnl-truth-distribution.png")


if __name__ == "__main__":
    print("Conda environment:", os.environ["CONDA_DEFAULT_ENV"])
    print("Python executable:", sys.executable)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")

    sys.exit(run())
