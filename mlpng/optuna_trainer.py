#!/usr/bin/env python3
"""Optuna-based trainer dedicated to tuning the Healpy U-Net."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import optuna
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
    HealpyPseudoConv_Transpose,
)

from mlpng import Core
import mlpng.core
from mlpng.utils import setup_logging
from mlpng.utils.dataloaders import KappaDataset
from mlpng.utils.callbacks import RMSELoss

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LOGGER = setup_logging("mlpng.optuna_trainer", level=logging.INFO)
logging.getLogger("mlpng.core").setLevel(logging.INFO)
logging.getLogger("mlpng.utils.dataloaders").setLevel(logging.INFO)

DATA_FRACTION = 0.01
BASE_SPLIT = np.array([0.8, 0.1, 0.1], dtype=np.float32)
DEFAULT_DUPLICATES = [10, 5, 2]
CACHE_PREFIX = "optuna-unet"
ACTIVATIONS = {
    "gelu": tf.keras.activations.gelu,
    "relu": tf.keras.activations.relu,
    "sigmoid": tf.keras.activations.sigmoid,
}

_DATASET_CACHE: dict[str, tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]] = {}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "settings_file", help="Path to the settings JSON consumed by Core"
    )
    parser.add_argument(
        "--trials", type=int, default=500, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional wall-clock timeout (seconds)",
    )
    parser.add_argument(
        "--study-name", default="optuna-unet", help="Name of the Optuna study"
    )
    parser.add_argument(
        "--storage", default=None, help="Optuna storage URL for persisting studies"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=30, help="Maximum epochs per trial"
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Early stopping patience"
    )
    parser.add_argument(
        "--pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Optuna pruning based on validation loss",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Train a final model with the best hyper-parameters and save it",
    )
    parser.add_argument(
        "--best-model-path",
        default=None,
        help="Custom path for the exported best model (.keras)",
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def build_run_name(core: Core, prefix: str = "optuna") -> str:
    timestamp = int(time.time())
    job = core.slurm.job if core.slurm.job else "local"
    return f"{prefix}-{core.shapes_str()}-{job}-{timestamp}"


def configure_strategy() -> tf.distribute.Strategy:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to set memory growth on %s: %s", gpu, exc)
        if len(gpus) > 1:
            LOGGER.info("Using MirroredStrategy across %d GPUs", len(gpus))
            return tf.distribute.MirroredStrategy()
    LOGGER.info("Defaulting to the global TF strategy")
    return tf.distribute.get_strategy()


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nside,
        npix,
        fin,
        fout,
        activation,
        max_batch_size,
        k_order,
        use_multik,
        use_bn=True,
        use_bias=False,
        pool=True,
        pool_type="MAX",
        layer_depth=2,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        indices = np.arange(npix)
        layers = []
        for depth_idx in range(max(1, layer_depth)):
            if use_multik and depth_idx > 0:
                k_val = 1
            else:
                k_val = k_order
            layers.append(
                HealpyChebyshev(
                    K=k_val,
                    Fout=fout,
                    activation=activation,
                    use_bn=use_bn,
                    use_bias=use_bias,
                )
            )
        if dropout_rate > 0.0:
            layers.append(Dropout(dropout_rate))

        self.body = HealpyGCNN(
            nside=nside,
            indices=indices,
            layers=layers,
            n_neighbors=8,
            max_batch_size=max_batch_size,
            initial_Fin=fin,
        )

        self.pool = None
        if pool:
            self.pool = HealpyGCNN(
                nside=nside,
                indices=indices,
                layers=[HealpyPool(1, pool_type)],
                n_neighbors=8,
                max_batch_size=max_batch_size,
                initial_Fin=fin,
            )

    def call(self, inputs, training=False):  # type: ignore[override]
        x = skip = self.body(inputs, training=training)
        if self.pool is not None:
            x = self.pool(x, training=training)
        return x, skip


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        nside,
        npix,
        fin,
        fout,
        activation,
        k_order,
        use_multik,
        max_batch_size,
        upsample=True,
        use_bn=True,
        use_bias=False,
        layer_depth=2,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        indices = np.arange(npix)
        layers = []
        if upsample:
            layers.append(HealpyPseudoConv_Transpose(1, fout))

        for depth_idx in range(layer_depth):
            if use_multik and depth_idx > 0:
                k_val = 1
            else:
                k_val = k_order
            layers.append(
                HealpyChebyshev(
                    K=k_val,
                    Fout=fout,
                    activation=activation,
                    use_bn=use_bn,
                    use_bias=use_bias,
                )
            )

        if dropout_rate > 0.0:
            layers.append(Dropout(dropout_rate))

        self.body = HealpyGCNN(
            nside=nside,
            indices=indices,
            layers=layers,
            n_neighbors=8,
            max_batch_size=max_batch_size,
            initial_Fin=fin,
        )

    def call(self, inputs, skip, training=False):  # type: ignore[override]
        x = self.body(inputs, training=training)
        if skip is not None:
            x = x + skip
        return x


class TunableResidualHealpyUNet:
    def __init__(
        self,
        input_shape,
        activation_name="gelu",
        max_batch_size=32,
        dropout_rate=0.1,
        use_multik=False,
        use_bn=True,
        use_input_bn=True,
        use_bias=True,
        k_step=3,
        initial_k=3,
        pool_type="MAX",
        encoder_depth=2,
        decoder_depth=2,
        bottleneck_depth=2,
    ):
        self.input_shape = input_shape
        activation_key = activation_name.lower()
        if activation_key not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation_name}'")
        self.activation = ACTIVATIONS[activation_key]
        self.max_batch_size = max_batch_size
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.use_input_bn = use_input_bn
        self.use_bias = use_bias
        self.k_step = k_step
        self.initial_k = initial_k
        self.use_multik = use_multik
        pool_mode = pool_type.upper()
        if pool_mode not in {"MAX", "AVG"}:
            raise ValueError("pool_type must be either 'MAX' or 'AVG'")
        self.pool_type = pool_mode
        self.encoder_depth = max(1, encoder_depth)
        self.decoder_depth = max(1, decoder_depth)
        self.bottleneck_depth = max(1, bottleneck_depth)
        self.npix = input_shape[1]
        self.npol = input_shape[2]
        self.nside = hp.npix2nside(self.npix)

    def _channel_schedule(self):
        depth = int(math.log2(self.nside))
        base = [self.npol]
        for i in range(depth + 1):
            base_width = 2 ** (i + 4)
            width = int(round(base_width))
            base.append(width)
        print("Channel schedule:", base)
        return base[: depth + 2]

    def _k_schedule(self):
        depth = int(math.log2(self.nside))
        ks = []
        for i in range(depth + 1):
            base_k = self.initial_k + 2 * (i // self.k_step)
            ks.append(base_k)
        print("K schedule:", ks)
        return ks

    def get_model(self) -> tf.keras.Model:
        channels = self._channel_schedule()
        ks = self._k_schedule()
        depth = len(ks) - 1
        level_npixels = [self.npix // (4**i) for i in range(depth + 1)]
        level_nsides = [hp.npix2nside(npix) for npix in level_npixels]

        inputs = tf.keras.Input(shape=self.input_shape[1:])
        x = BatchNormalization()(inputs) if self.use_input_bn else inputs
        skips = []

        for i in range(depth):
            block = EncoderBlock(
                level_nsides[i],
                level_npixels[i],
                fin=channels[i],
                fout=channels[i + 1],
                activation=self.activation,
                max_batch_size=self.max_batch_size,
                k_order=ks[i],
                use_multik=self.use_multik,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                use_bn=self.use_bn,
                pool_type=self.pool_type,
                layer_depth=self.encoder_depth,
            )
            x, skip = block(x)
            skips.append(skip)

        bottleneck_layers = []
        for depth_idx in range(max(1, self.bottleneck_depth)):
            if self.use_multik and depth_idx > 0:
                k_val = 1
            else:
                k_val = ks[-1]
            bottleneck_layers.append(
                HealpyChebyshev(
                    K=k_val,
                    Fout=channels[-1],
                    activation=self.activation,
                    use_bn=self.use_bn,
                    use_bias=self.use_bias,
                )
            )

        bottleneck = HealpyGCNN(
            nside=level_nsides[-1],
            indices=np.arange(level_npixels[-1]),
            layers=bottleneck_layers,
            n_neighbors=8,
            max_batch_size=self.max_batch_size,
            initial_Fin=channels[-1],
            name="bottleneck",
        )
        x = bottleneck(x)

        for i in reversed(range(1, depth + 1)):
            block = DecoderBlock(
                level_nsides[i],
                level_npixels[i],
                fin=channels[i],
                fout=channels[i],
                activation=self.activation,
                max_batch_size=self.max_batch_size,
                k_order=ks[i],
                use_multik=self.use_multik,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                use_bn=self.use_bn,
                layer_depth=self.decoder_depth,
            )
            x = block(x, skips[i - 1])

        head = DecoderBlock(
            level_nsides[0],
            level_npixels[0],
            fin=channels[i],
            fout=channels[0],
            activation=self.activation,
            max_batch_size=self.max_batch_size,
            k_order=ks[0],
            use_multik=self.use_multik,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            use_bn=self.use_bn,
            layer_depth=self.decoder_depth,
        )
        x = head(x, None)

        output_layer = HealpyGCNN(
            nside=self.nside,
            indices=np.arange(self.npix),
            layers=[
                HealpyChebyshev(
                    K=1,
                    Fout=self.npol,
                    activation=None,
                    use_bn=False,
                    use_bias=True,
                )
            ],
            n_neighbors=8,
            max_batch_size=self.max_batch_size,
            initial_Fin=self.npol,
            name="output",
        )
        outputs = output_layer(x)
        return tf.keras.Model(inputs, outputs, name="tunable_residual_healpy_unet")


def prepare_datasets(
    core: Core,
    batch_size: int,
    duplicates: list[int],
    cache_tag: str,
    kappa_scale: float = 1.0,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    cache_key = f"{cache_tag}-bs{batch_size}-ks{kappa_scale}"
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    ds = KappaDataset.fromCore(
        core, x_output="lensed", y_output="kappa", kappa_scale=kappa_scale
    )
    fractions = BASE_SPLIT * DATA_FRACTION
    cache_dir = os.path.join(core.dirs["base"], "tf-cache")
    os.makedirs(cache_dir, exist_ok=True)
    available_cpus = max(1, core.slurm.n_cpus or os.cpu_count() or 4)

    train, val, test = ds.split(
        train_size=float(fractions[0]),
        val_size=float(fractions[1]),
        test_size=float(fractions[2]),
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        cache_dir=cache_dir,
        cache_file=cache_tag,
        gen_batch_size=available_cpus,
    )

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train.map(lambda _x, y: y))

    def normalize_y(x, y):
        return x, normalizer(y)

    def finalize(ds_split, name, warm=False):
        cached = (
            ds_split.map(normalize_y, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        if warm:
            for _ in cached:
                pass
            LOGGER.debug("Dataset %s cached", name)
        return cached

    train = finalize(train, "train", warm=True)
    val = finalize(val, "val", warm=True)
    test = finalize(test, "test")
    _DATASET_CACHE[cache_key] = (train, val, test)
    return train, val, test


def estimate_steps(core: Core, batch_size: int) -> tuple[int, int]:
    train_fraction = float(BASE_SPLIT[0] * DATA_FRACTION)
    approx_samples = (
        max(1, int(core.total_sims * train_fraction)) * DEFAULT_DUPLICATES[0]
    )
    steps = max(1, math.ceil(approx_samples / batch_size))
    return steps, steps * 2


@dataclass
class ObjectiveContext:
    core: Core
    strategy: tf.distribute.Strategy
    args: argparse.Namespace
    run_name: str


class UnetObjective:
    def __init__(self, ctx: ObjectiveContext):
        self.ctx = ctx
        self.cache_tag = f"{CACHE_PREFIX}-{ctx.core.name}"

    def __call__(self, trial: optuna.Trial) -> float:
        # K.clear_session()
        batch_size = 32  # trial.suggest_categorical("batch_size", [16, 24, 32, 64])
        dropout = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.1)
        initial_k = trial.suggest_categorical("initial_k", [1, 2, 3])
        k_step = trial.suggest_categorical("k_step", [1, 2, 3])
        multik = trial.suggest_categorical("use_multik", [False, True])
        encoder_depth = trial.suggest_categorical("encoder_depth", [1, 2, 3])
        decoder_depth = trial.suggest_categorical("decoder_depth", [1, 2, 3])
        bottleneck_depth = trial.suggest_categorical("bottleneck_depth", [1, 2, 3])
        pool_type = trial.suggest_categorical("pool_type", ["MAX", "AVG"])
        activation_name = trial.suggest_categorical(
            "activation", ["gelu", "relu", "sigmoid"]
        )
        use_bn = trial.suggest_categorical("use_bn", [False, True])
        use_input_bn = trial.suggest_categorical("use_input_bn", [False, True])
        use_bias = trial.suggest_categorical("use_bias", [False, True])
        initial_lr = trial.suggest_float("initial_lr", 5e-5, 2e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        decay_rate = trial.suggest_float("lr_decay", 0.92, 0.99, step=0.01)
        use_ema = trial.suggest_categorical("use_ema", [False, True])
        use_amsgrad = trial.suggest_categorical("use_amsgrad", [False, True])
        kappa_scale = trial.suggest_categorical("kappa_scale", [1.0, np.sqrt(1e7)])

        train_ds, val_ds, _ = prepare_datasets(
            self.ctx.core,
            batch_size=batch_size,
            duplicates=DEFAULT_DUPLICATES,
            cache_tag=self.cache_tag,
            kappa_scale=kappa_scale,
        )

        steps, decay_steps = estimate_steps(self.ctx.core, batch_size)
        with self.ctx.strategy.scope():
            lr_schedule = ExponentialDecay(
                initial_learning_rate=initial_lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,
            )
            model = TunableResidualHealpyUNet(
                (None, self.ctx.core.npix, self.ctx.core.npols),
                activation_name=activation_name,
                max_batch_size=batch_size,
                dropout_rate=dropout,
                k_step=k_step,
                initial_k=initial_k,
                use_multik=multik,
                pool_type=pool_type,
                encoder_depth=encoder_depth,
                decoder_depth=decoder_depth,
                bottleneck_depth=bottleneck_depth,
                use_bn=use_bn,
                use_input_bn=use_input_bn,
                use_bias=use_bias,
            ).get_model()
            optimizer = AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                amsgrad=use_amsgrad,
                use_ema=use_ema,
            )
            model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=[],
            )

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss",
                patience=self.ctx.args.patience,
                restore_best_weights=True,
            ),
        ]
        if self.ctx.args.pruning:
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.ctx.args.max_epochs,
            steps_per_epoch=steps,
            callbacks=callbacks,
            verbose=0,
        )
        val_loss = min(history.history["val_loss"])
        return float(val_loss)


def main() -> int:
    args, unknown = parse_args()
    core = Core(argv=[args.settings_file] + unknown)
    strategy = configure_strategy()
    run_name = build_run_name(core)
    LOGGER.info("Run name: %s", run_name)
    ctx = ObjectiveContext(
        core=core,
        strategy=strategy,
        args=args,
        run_name=run_name,
    )

    LOGGER.info("Using %.2f%% of the data for tuning", DATA_FRACTION * 100)

    storage_url = args.storage
    load_existing = True  # bool(storage_url)
    if not storage_url:
        storage_dir = Path("data/tuner")
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_file = storage_dir / f"optuna.db"
        storage_url = f"sqlite:///{storage_file}"
        LOGGER.info("Optuna storage set to %s", storage_file)

    sampler = TPESampler(multivariate=True, group=True, seed=ctx.core.seed)
    pruner = MedianPruner(n_warmup_steps=10) if args.pruning else None
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_existing,
    )

    objective = UnetObjective(ctx)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    LOGGER.info("Best value: %.6f", study.best_value)
    LOGGER.info("Best params: %s", study.best_params)


if __name__ == "__main__":
    sys.exit(main())
