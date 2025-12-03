#!/usr/bin/env python3
"""Optuna-based trainer for tuning the CNN-Transformer Encoder fnl model."""

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
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import HealpyChebyshev, HealpyPool, Healpy_Transformer

from mlpng import Core
import mlpng.core
from mlpng.utils import setup_logging, rmse_metrics
from mlpng.utils.dataloaders import KappaDataset

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LOGGER = setup_logging("mlpng.optuna_trainer", level=logging.INFO)
logging.getLogger("mlpng.core").setLevel(logging.INFO)
logging.getLogger("mlpng.utils.dataloaders").setLevel(logging.INFO)

DATA_FRACTION = 0.1
BASE_SPLIT = np.array([0.8, 0.1, 0.1], dtype=np.float32)
DEFAULT_DUPLICATES = [10, 2, 2]
CACHE_PREFIX = "optuna-encoder-fnl"

_DATASET_CACHE: dict[str, tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]] = {}


def compute_max_depths(nside: int) -> dict[int, int]:
    """Precompute maximum encoder depth for each pool_p value."""
    return {
        1: int(math.log(nside, 2)),  # pool_p=1 -> nside_factor=2
        2: int(math.log(nside, 4)),  # pool_p=2 -> nside_factor=4
    }


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
        "--study-name", default="encoder-fnl-n256", help="Name of the Optuna study"
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
    return f"{prefix}-n{core.nside}-{core.shapes_str()}-{job}-{timestamp}"


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
    """Encoder block with optional Healpy Transformer attention."""

    def __init__(
        self,
        nside,
        npix,
        fin,
        fout,
        K,
        pool_p,
        max_batch_size,
        dropout_rate=0.1,
        use_transformer=False,
        num_heads=8,
        transformer_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nside, self.npix, self.fin, self.fout = nside, npix, fin, fout
        self.K, self.pool_p, self.max_batch_size, self.dropout_rate = (
            K,
            pool_p,
            max_batch_size,
            dropout_rate,
        )

        layers = [
            HealpyChebyshev(
                K=K,
                Fout=fout,
                activation="gelu",
                use_bn=True,
                use_bias=True,
                initializer=tf.keras.initializers.HeNormal(),
            )
        ]
        if use_transformer:
            key_dim = max(1, min(1024, fout // num_heads))
            layers.append(
                Healpy_Transformer(
                    key_dim=key_dim,
                    num_heads=num_heads,
                    positional_encoding=True,
                    n_layers=transformer_layers,
                    activation="gelu",
                    layer_norm=True,
                )
            )
        layers.extend(
            [
                Dropout(dropout_rate),
                HealpyPool(pool_p, "AVG"),
            ]
        )

        self.body = HealpyGCNN(
            nside=nside,
            indices=np.arange(npix),
            layers=layers,
            n_neighbors=8,
            max_batch_size=max_batch_size,
            initial_Fin=fin,
        )

    def call(self, x, training=False):
        return self.body(x, training=training)


class TunableEncoderFnlModel:
    """CNN-Transformer encoder model for fnl prediction with tunable architecture."""

    def __init__(
        self,
        input_shape,
        n_outputs,
        max_batch_size=32,
        pool_p=2,
        transformer_levels=0,
        num_heads=8,
        transformer_layers=2,
        dropout_rate=0.1,
        dense_units=64,
        dense_layers=2,
        head_dropout=0.0,
    ):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.max_batch_size = max_batch_size
        self.pool_p = pool_p
        self.transformer_levels = (
            transformer_levels  # int: number of deepest levels with transformers
        )
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.dense_layers = dense_layers
        self.head_dropout = head_dropout
        self.npix = input_shape[1]
        self.nside = hp.npix2nside(self.npix)
        self.npol = input_shape[2]
        self.initializer = tf.keras.initializers.HeNormal()

    def get_model(self) -> tf.keras.Model:
        nside_factor = 2**self.pool_p
        pixel_factor = 4**self.pool_p

        depth = int(math.log(self.nside, nside_factor))
        level_nsides = [self.nside // (nside_factor**i) for i in range(depth + 1)]
        level_npixels = [12 * ns**2 for ns in level_nsides]
        channels = [self.npol] + [2 ** (i + 5) for i in range(depth + 1)]
        Ks = [3] * (depth + 1)

        # transformer_levels is an int: 0 = none, depth = all
        # Apply transformers to the last `transformer_levels` encoder blocks
        use_transformer = [False] * depth
        if self.transformer_levels > 0:
            start_idx = max(0, depth - self.transformer_levels)
            for i in range(start_idx, depth):
                use_transformer[i] = True

        LOGGER.debug(
            "Depth: %d, pool_p: %d (%dx pixel reduction)",
            depth,
            self.pool_p,
            pixel_factor,
        )
        LOGGER.debug("Level nsides: %s", level_nsides)
        LOGGER.debug("Level npixels: %s", level_npixels)
        LOGGER.debug("Channels: %s, Ks: %s", channels, Ks)
        LOGGER.debug(
            "Transformer at levels: %s",
            [i for i, t in enumerate(use_transformer) if t],
        )

        inputs = tf.keras.Input(shape=self.input_shape[1:], name="lensed")
        x = inputs

        for i in range(depth):
            x = EncoderBlock(
                level_nsides[i],
                level_npixels[i],
                fin=channels[i],
                fout=channels[i + 1],
                K=Ks[i],
                pool_p=self.pool_p,
                max_batch_size=self.max_batch_size,
                dropout_rate=self.dropout_rate,
                use_transformer=use_transformer[i],
                num_heads=self.num_heads,
                transformer_layers=self.transformer_layers,
            )(x)

        # Dense head
        x = Flatten()(x)
        for _ in range(self.dense_layers):
            x = Dense(
                self.dense_units,
                activation="gelu",
                kernel_initializer=self.initializer,
            )(x)
            if self.head_dropout > 0:
                x = Dropout(self.head_dropout)(x)

        outputs = Dense(self.n_outputs, kernel_initializer=self.initializer)(x)

        return tf.keras.Model(inputs, outputs, name="tunable_encoder_fnl")


def prepare_datasets(
    core: Core,
    batch_size: int,
    duplicates: list[int],
    cache_tag: str,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    cache_key = f"{cache_tag}-bs{batch_size}"
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    ds = KappaDataset.fromCore(core, x_output="lensed", y_output="fnl")
    fractions = BASE_SPLIT * DATA_FRACTION
    cache_dir = os.environ.get("SCRATCH", "/tmp") + "/tf_cache"
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
    max_depths: dict[int, int]


class EncoderFnlObjective:
    """Optuna objective for tuning the CNN-Transformer Encoder fnl model."""

    def __init__(self, ctx: ObjectiveContext):
        self.ctx = ctx
        self.cache_tag = f"{CACHE_PREFIX}-{ctx.core.name}"

    def __call__(self, trial: optuna.Trial) -> float:
        batch_size = 32

        # Learning rate schedule selection
        use_cosine_decay = trial.suggest_categorical("use_cosine_decay", [True, False])
        initial_lr = trial.suggest_float("initial_lr", 1e-5, 1e-1, log=True)

        # Architecture: pool_p first to determine depth
        pool_p = trial.suggest_categorical("pool_p", [1, 2])
        max_depth = self.ctx.max_depths[pool_p]

        # Transformer configuration
        transformer_levels = trial.suggest_int("transformer_levels", 0, max_depth)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        transformer_layers = trial.suggest_categorical("transformer_layers", [1, 2, 3])

        # Regularization
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.05)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

        # Dense head configuration
        dense_units = trial.suggest_categorical("dense_units", [32, 64, 128, 256])
        dense_layers = trial.suggest_categorical("dense_layers", [1, 2, 3])
        head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5, step=0.1)

        train_ds, val_ds, _ = prepare_datasets(
            self.ctx.core,
            batch_size=batch_size,
            duplicates=DEFAULT_DUPLICATES,
            cache_tag=self.cache_tag,
        )

        steps, decay_steps = estimate_steps(self.ctx.core, batch_size)

        with self.ctx.strategy.scope():
            if use_cosine_decay:
                lr_schedule = CosineDecayRestarts(
                    initial_learning_rate=initial_lr,
                    first_decay_steps=decay_steps,
                    t_mul=2.0,
                    m_mul=0.95,
                    alpha=0.01,
                )
            else:
                lr_schedule = ExponentialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=decay_steps,
                    decay_rate=0.96,
                    staircase=True,
                )

            model = TunableEncoderFnlModel(
                (None, self.ctx.core.npix, self.ctx.core.npols),
                n_outputs=len(self.ctx.core.shapes),
                max_batch_size=batch_size,
                pool_p=pool_p,
                transformer_levels=transformer_levels,
                num_heads=num_heads,
                transformer_layers=transformer_layers,
                dropout_rate=dropout_rate,
                dense_units=dense_units,
                dense_layers=dense_layers,
                head_dropout=head_dropout,
            ).get_model()

            optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
            model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=rmse_metrics(self.ctx.core.shapes),
            )

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss",
                patience=self.ctx.args.patience,
                restore_best_weights=True,
            ),
        ]

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
    max_depths = compute_max_depths(core.nside)

    LOGGER.info("Run name: %s", run_name)
    LOGGER.info("Max depths by pool_p: %s", max_depths)

    ctx = ObjectiveContext(
        core=core,
        strategy=strategy,
        args=args,
        run_name=run_name,
        max_depths=max_depths,
    )

    LOGGER.info("Using %.2f%% of the data for tuning", DATA_FRACTION * 100)

    storage_url = args.storage
    load_existing = True
    if not storage_url:
        storage_dir = Path("data/tuner")
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_file = storage_dir / "optuna.db"
        storage_url = f"sqlite:///{storage_file}"
        LOGGER.info("Optuna storage set to %s", storage_file)

    sampler = TPESampler(multivariate=True, seed=ctx.core.seed, constant_liar=True)
    pruner = MedianPruner(n_warmup_steps=10) if args.pruning else None
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=storage_url,
        sampler=sampler,
        load_if_exists=load_existing,
    )

    objective = EncoderFnlObjective(ctx)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    LOGGER.info("Best value: %.6f", study.best_value)
    LOGGER.info("Best params: %s", study.best_params)

    return 0


if __name__ == "__main__":
    sys.exit(main())
