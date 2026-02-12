#!/usr/bin/env python3
"""Optuna-based trainer for tuning the CNN-Transformer Encoder fnl model."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Suppress ALL TensorFlow/CUDA/XLA logging BEFORE any imports
# =============================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Suppress XLA and CUDA factory registration messages
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Redirect C++ logs to /dev/null
os.environ["TF_CPP_LOG_THREAD_ID"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

warnings.filterwarnings("ignore")

# Suppress absl logging before TensorFlow import
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.ERROR)

# =============================================================================
# CONFIGURATION - Training hyperparameters (not CLI args)
# =============================================================================
CONFIG = {
    "max_epochs": 30,  # Maximum epochs per trial
    "patience": 8,  # Early stopping patience
    "pruning": False,  # Enable Optuna pruning
    "batch_size": 8,  # Training batch size
    "data_fraction": 1.0,  # Fraction of data for tuning
}
# =============================================================================

import healpy as hp
import numpy as np
import optuna
import tensorflow as tf
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import HealpyChebyshev, HealpyPool

from mlpng import Core
from mlpng.utils import setup_logging, rmse_metrics
from mlpng.utils.dataloaders import KappaDataset

# Suppress TensorFlow loggers after import
for logger_name in [
    "tensorflow",
    "tensorflow.compiler",
    "tensorflow.compiler.tf2tensorrt",
    "tensorflow.python",
    "absl",
    "h5py",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)


def create_sigma_weighted_loss(sigma_all):
    """
    Create a custom MSE loss function weighted by inverse sigma (precision weighting).

    Each output is normalized by its corresponding sigma value, so outputs with
    smaller sigma (higher precision) have their errors weighted more heavily.
    This is equivalent to maximum likelihood estimation for Gaussian errors.

    Args:
        sigma_all: Array of sigma values, one per output shape

    Returns:
        Loss function that takes (y_true, y_pred) and returns weighted MSE
    """
    sigma_tensor = tf.constant(sigma_all, dtype=tf.float32)

    def sigma_weighted_mse(y_true, y_pred):
        # Compute squared error per output
        squared_error = tf.square(y_true - y_pred)  # Shape: (batch_size, n_outputs)

        # Weight by inverse variance (1 / sigma^2)
        # Outputs with smaller sigma get higher weight
        weights = 1.0 / (sigma_tensor**2)

        # Apply weights and compute mean
        weighted_loss = squared_error * weights
        return tf.reduce_mean(weighted_loss)

    return sigma_weighted_mse


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments. Unknown args are forwarded to Core."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "settings_file", help="Path to the settings JSON consumed by Core"
    )
    parser.add_argument(
        "--trials", type=int, default=500, help="Number of Optuna trials to run"
    )
    return parser.parse_known_args()


BASE_SPLIT = np.array([0.8, 0.1, 0.1], dtype=np.float32)
DEFAULT_DUPLICATES = [10, 10, 2]
CACHE_PREFIX = "all-full_ds-30epoch-2"

_DATASET_CACHE: dict[str, tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]] = {}


def compute_max_depths(nside: int) -> dict[int, int]:
    """Precompute maximum encoder depth for each pool_p value."""
    return {
        1: int(math.log(nside, 2)),  # pool_p=1 -> nside_factor=2
        2: int(math.log(nside, 4)),  # pool_p=2 -> nside_factor=4
    }


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
    """Encoder block with Healpy Chebyshev convolution and pooling."""

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
        encoder_activation="gelu",
        pool_type="AVG",
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

        # Select initializer based on activation
        if encoder_activation == "relu":
            initializer = tf.keras.initializers.HeNormal()
        else:  # gelu or sigmoid
            initializer = tf.keras.initializers.GlorotNormal()

        layers = [
            HealpyChebyshev(
                K=K,
                Fout=fout,
                activation=encoder_activation,
                use_bn=True,
                use_bias=True,
                initializer=initializer,
            ),
            Dropout(dropout_rate),
            HealpyPool(pool_p, pool_type),
        ]

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


class SplitHeadLayer(tf.keras.layers.Layer):
    """Split head layer with shared and shape-specific branches."""

    def __init__(
        self,
        n_outputs,
        shared_units=64,
        shared_layers=1,
        split_units=32,
        split_layers=1,
        head_activation="gelu",
        head_dropout=0.0,
        head_initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.shared_units = shared_units
        self.shared_layers = shared_layers
        self.split_units = split_units
        self.split_layers = split_layers
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_initializer = (
            head_initializer
            if head_initializer is not None
            else tf.keras.initializers.GlorotNormal()
        )

        # Build shared layers
        self.shared = []
        for _ in range(shared_layers):
            self.shared.append(
                Dense(
                    shared_units,
                    activation=head_activation,
                    kernel_initializer=self.head_initializer,
                )
            )
            if head_dropout > 0:
                self.shared.append(Dropout(head_dropout))

        # Build split branches (one per output)
        self.split_branches = []
        for i in range(n_outputs):
            branch = []
            for _ in range(split_layers):
                branch.append(
                    Dense(
                        split_units,
                        activation=head_activation,
                        kernel_initializer=self.head_initializer,
                        name=f"split_dense_{i}_layer_{_}",
                    )
                )
                if head_dropout > 0:
                    branch.append(
                        Dropout(head_dropout, name=f"split_dropout_{i}_layer_{_}")
                    )
            # Final output layer for this branch
            branch.append(
                Dense(1, kernel_initializer=self.head_initializer, name=f"output_{i}")
            )
            self.split_branches.append(branch)

    def call(self, x, training=False):
        # Pass through shared layers
        for layer in self.shared:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        # Split into separate branches
        outputs = []
        for branch in self.split_branches:
            branch_out = x
            for layer in branch:
                if isinstance(layer, Dropout):
                    branch_out = layer(branch_out, training=training)
                else:
                    branch_out = layer(branch_out)
            outputs.append(branch_out)

        # Concatenate all outputs
        return Concatenate()(outputs)


class TunableEncoderFnlModel:
    """CNN encoder model for fnl prediction with tunable architecture."""

    def __init__(
        self,
        input_shape,
        n_outputs,
        max_batch_size=32,
        pool_p=2,
        K=3,
        encoder_activation="gelu",
        head_activation="gelu",
        pool_type="AVG",
        dropout_rate=0.1,
        dense_units=64,
        dense_layers=2,
        head_dropout=0.0,
        use_split_head=False,
        shared_units=64,
        shared_layers=1,
        split_units=32,
        split_layers=1,
    ):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.max_batch_size = max_batch_size
        self.pool_p = pool_p
        self.K = K
        self.encoder_activation = encoder_activation
        self.head_activation = head_activation
        self.pool_type = pool_type
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.dense_layers = dense_layers
        self.head_dropout = head_dropout
        self.use_split_head = use_split_head
        self.shared_units = shared_units
        self.shared_layers = shared_layers
        self.split_units = split_units
        self.split_layers = split_layers
        self.npix = input_shape[1]
        self.nside = hp.npix2nside(self.npix)
        self.npol = input_shape[2]

        # Select head initializer based on activation
        if head_activation == "relu":
            self.head_initializer = tf.keras.initializers.HeNormal()
        else:  # gelu or sigmoid
            self.head_initializer = tf.keras.initializers.GlorotNormal()

    def get_model(self) -> tf.keras.Model:
        nside_factor = 2**self.pool_p

        depth = int(math.log(self.nside, nside_factor))
        level_nsides = [self.nside // (nside_factor**i) for i in range(depth + 1)]
        level_npixels = [12 * ns**2 for ns in level_nsides]
        channels = [self.npol] + [2 ** (i + 5) for i in range(depth + 1)]
        Ks = [self.K] * depth

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
                encoder_activation=self.encoder_activation,
                pool_type=self.pool_type,
            )(x)

        # Dense head or split head
        x = Flatten()(x)

        if self.use_split_head:
            # Use split head with shared and shape-specific branches
            outputs = SplitHeadLayer(
                n_outputs=self.n_outputs,
                shared_units=self.shared_units,
                shared_layers=self.shared_layers,
                split_units=self.split_units,
                split_layers=self.split_layers,
                head_activation=self.head_activation,
                head_dropout=self.head_dropout,
                head_initializer=self.head_initializer,
            )(x)
        else:
            # Use standard dense head
            for _ in range(self.dense_layers):
                x = Dense(
                    self.dense_units,
                    activation=self.head_activation,
                    kernel_initializer=self.head_initializer,
                )(x)
                if self.head_dropout > 0:
                    x = Dropout(self.head_dropout)(x)

            outputs = Dense(self.n_outputs, kernel_initializer=self.head_initializer)(x)

        return tf.keras.Model(inputs, outputs, name="tunable_encoder_fnl")


def prepare_datasets(
    core: Core,
    batch_size: int,
    duplicates: list[int],
    cache_tag: str,
    data_fraction: float,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    cache_key = f"{cache_tag}-bs{batch_size}"
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    ds = KappaDataset.fromCore(core, x_output="lensed", y_output="fnl")
    fractions = BASE_SPLIT * data_fraction
    available_cpus = 8  # max(1, core.slurm.n_cpus or os.cpu_count() or 4)

    # Use in-memory caching to avoid concurrent file-based cache lockfile conflicts
    # when multiple Optuna trials run in parallel
    train, val, test = ds.split(
        train_size=float(fractions[0]),
        val_size=float(fractions[1]),
        test_size=float(fractions[2]),
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        buffer_size=8,
        cache_file="",  # Empty string for in-memory caching
        gen_batch_size=available_cpus,
    )
    _DATASET_CACHE[cache_key] = (train, val, test)
    return train, val, test


def estimate_steps(
    core: Core, batch_size: int, data_fraction: float
) -> tuple[int, int]:
    train_fraction = float(BASE_SPLIT[0] * data_fraction)
    approx_samples = (
        max(1, int(core.total_sims * train_fraction)) * DEFAULT_DUPLICATES[0]
    )
    steps = max(1, math.ceil(approx_samples / batch_size))
    return steps, steps * 2


@dataclass
class ObjectiveContext:
    core: Core
    strategy: tf.distribute.Strategy
    config: dict
    run_name: str
    max_depths: dict[int, int]


class EncoderFnlObjective:
    """Optuna objective for tuning the CNN Encoder fnl model."""

    def __init__(self, ctx: ObjectiveContext):
        self.ctx = ctx
        self.cache_tag = f"{CACHE_PREFIX}-{ctx.core.name}-n{ctx.core.nside}"
        # Create custom loss with sigma values baked in
        sigma_all = ctx.core.get_likelihoods(True)
        self.custom_loss = create_sigma_weighted_loss(sigma_all)
        LOGGER.info("Sigma values for shapes %s: %s", ctx.core.shapes, sigma_all)

    def __call__(self, trial: optuna.Trial) -> float:
        batch_size = self.ctx.config["batch_size"]
        nside = self.ctx.core.nside

        # Architecture: pool_p determines network depth
        pool_p = trial.suggest_categorical("pool_p", [1, 2])
        max_depth = self.ctx.max_depths[pool_p]

        # Kernel size for Chebyshev polynomials
        K = trial.suggest_categorical("K", [2, 3, 5])

        # Activation functions
        encoder_activation = trial.suggest_categorical(
            "encoder_activation", ["relu", "gelu", "sigmoid"]
        )
        head_activation = trial.suggest_categorical(
            "head_activation", ["relu", "gelu", "sigmoid"]
        )

        # Pooling type
        pool_type = trial.suggest_categorical("pool_type", ["AVG", "MAX"])

        # Learning rate
        use_cosine_decay = trial.suggest_categorical("use_cosine_decay", [True, False])
        initial_lr = trial.suggest_float("initial_lr", 1e-6, 1e-2, log=True)

        # Decay rate for both cosine and exponential decay
        decay_rate = trial.suggest_float("decay_rate", 0.9, 0.98, step=0.01)

        # Regularization
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.05)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)

        # Head configuration - always use split head with all parameters tuned
        use_split_head = True

        # Split head hyperparameters - always tuned
        shared_units = trial.suggest_categorical("shared_units", [32, 64, 128])
        dense_layers = trial.suggest_int("dense_layers", 0, 3)  # Shared layers
        split_units = trial.suggest_categorical("split_units", [16, 32, 64])
        head_layers = trial.suggest_int("head_layers", 0, 3)  # Per-shape branch layers
        head_dropout = trial.suggest_float("head_dropout", 0.0, 0.3, step=0.1)

        # Aliases for compatibility
        shared_layers = dense_layers  # Use dense_layers as shared_layers
        split_layers = head_layers  # Use head_layers as split_layers

        train_ds, val_ds, _ = prepare_datasets(
            self.ctx.core,
            batch_size=batch_size,
            duplicates=DEFAULT_DUPLICATES,
            cache_tag=self.cache_tag,
            data_fraction=self.ctx.config["data_fraction"],
        )

        steps, decay_steps = estimate_steps(
            self.ctx.core, batch_size, self.ctx.config["data_fraction"]
        )

        with self.ctx.strategy.scope():
            if use_cosine_decay:
                lr_schedule = CosineDecayRestarts(
                    initial_learning_rate=initial_lr,
                    first_decay_steps=decay_steps,
                    t_mul=2.0,
                    m_mul=decay_rate,
                    alpha=0.001,
                )
            else:
                lr_schedule = ExponentialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True,
                )

            model = TunableEncoderFnlModel(
                (None, self.ctx.core.npix, self.ctx.core.npols),
                n_outputs=len(self.ctx.core.shapes),
                max_batch_size=batch_size,
                pool_p=pool_p,
                K=K,
                encoder_activation=encoder_activation,
                head_activation=head_activation,
                pool_type=pool_type,
                dropout_rate=dropout_rate,
                dense_units=shared_units,
                dense_layers=dense_layers,
                head_dropout=head_dropout,
                use_split_head=use_split_head,
                shared_units=shared_units,
                shared_layers=dense_layers,
                split_units=split_units,
                split_layers=head_layers,
            ).get_model()

            optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
            model.compile(
                optimizer=optimizer,
                loss="mse",  # self.custom_loss,
                metrics=rmse_metrics(self.ctx.core.shapes),
            )

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss",
                patience=self.ctx.config["patience"],
                restore_best_weights=True,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.ctx.config["max_epochs"],
            steps_per_epoch=steps,
            callbacks=callbacks,
            verbose=0,
        )
        val_loss = min(history.history["val_loss"])

        # Store per-shape RMSE metrics in trial user attributes
        for shape in self.ctx.core.shapes:
            val_rmse_key = f"val_rmse_{shape}"
            if val_rmse_key in history.history:
                # Get the best (minimum) validation RMSE for this shape
                best_val_rmse = min(history.history[val_rmse_key])
                trial.set_user_attr(f"best_val_rmse_{shape}", float(best_val_rmse))

        return float(val_loss)


def main() -> int:
    args, unknown = parse_args()

    # Clean up any stale TensorFlow cache lockfiles from previous runs
    # This prevents "AlreadyExistsError" from concurrent cache access
    cache_dirs = [
        os.environ.get("SCRATCH", "/tmp") + "/tf_cache",
        "/lustre/smuexa01/client/users/stevensonb/tf_cache",  # HPC specific
    ]
    for cache_dir in cache_dirs:
        if os.path.isdir(cache_dir):
            for lockfile in Path(cache_dir).glob("*.lockfile"):
                try:
                    lockfile.unlink()
                    LOGGER.debug("Removed stale cache lockfile: %s", lockfile)
                except OSError as e:
                    LOGGER.debug("Could not remove lockfile %s: %s", lockfile, e)

    core = Core(argv=[args.settings_file] + unknown)
    strategy = configure_strategy()
    run_name = build_run_name(core)
    max_depths = compute_max_depths(core.nside)

    LOGGER.info("Run name: %s", run_name)
    LOGGER.info("Max depths by pool_p: %s", max_depths)

    ctx = ObjectiveContext(
        core=core,
        strategy=strategy,
        config=CONFIG,
        run_name=run_name,
        max_depths=max_depths,
    )

    LOGGER.info("Using %.2f%% of the data for tuning", CONFIG["data_fraction"] * 100)

    # Generate study name from nside and shapes
    shapes_str = "_".join(str(s) for s in ctx.core.shapes)
    study_name = f"neo-mse-n{ctx.core.nside}-{shapes_str}"

    # Storage in local SQLite database
    storage_dir = Path("data/tuner")
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_file = storage_dir / f"optuna-n{ctx.core.nside}-2.db"
    storage_url = f"sqlite:///{storage_file}"
    LOGGER.info("Optuna storage: %s", storage_file)

    sampler = TPESampler(multivariate=True, seed=ctx.core.seed, constant_liar=True)
    pruner = MedianPruner(n_warmup_steps=10) if CONFIG["pruning"] else None
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = EncoderFnlObjective(ctx)
    study.optimize(objective, n_trials=args.trials)

    LOGGER.info("Best value: %.6f", study.best_value)
    LOGGER.info("Best params: %s", study.best_params)

    return 0


if __name__ == "__main__":
    setup_logging(__name__, level=logging.INFO)
    logging.getLogger("mlpng.core").setLevel(logging.INFO)
    logging.getLogger("mlpng.utils.dataloaders").setLevel(logging.INFO)

    sys.exit(main())
