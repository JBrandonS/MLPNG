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


class TaskSpecificHeads(tf.keras.layers.Layer):
    """Shape-dependent heads with increasing capacity for local, equilateral, orthogonal."""

    SHAPE_TO_HEAD_INDEX = {
        "local": 0,
        "equilateral": 1,
        "orthogonal": 2,
    }

    HEAD_ARCHITECTURES = {
        0: [64, 32, 32, 1],
        1: [32, 128, 256, 256, 64, 32, 1],
        2: [16, 64, 64, 32, 32, 1],
    }

    def __init__(self, n_outputs, task_names, activation="relu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.task_names = (
            task_names if isinstance(task_names, (list, tuple)) else [task_names]
        )
        self.activation = activation

        init = (
            tf.keras.initializers.HeNormal()
            if activation == "relu"
            else tf.keras.initializers.GlorotNormal()
        )

        def build_head(sizes, prefix, head_dropout=0.0, head_activation=None):
            layers = []
            if head_dropout > 0:
                layers.append(Dropout(head_dropout, name=f"{prefix}_dropout_0"))
            act = head_activation or activation
            for i, size in enumerate(sizes[:-1]):
                layers.append(
                    Dense(
                        size,
                        activation=act,
                        kernel_initializer=init,
                        name=f"{prefix}_dense_{i+1}",
                    )
                )
            layers.append(
                Dense(sizes[-1], kernel_initializer=init, name=f"{prefix}_output")
            )
            return layers

        self.heads = {}
        for shape_name in self.task_names:
            if shape_name in self.SHAPE_TO_HEAD_INDEX:
                head_idx = self.SHAPE_TO_HEAD_INDEX[shape_name]
                head_arch = self.HEAD_ARCHITECTURES[head_idx]
                self.heads[shape_name] = build_head(
                    head_arch, f"head_{shape_name}", dropout, activation
                )
            else:
                LOGGER.warning(f"Unknown shape: {shape_name}, skipping head creation")

    def call(self, x, training=False):
        outputs = []
        for shape_name in self.task_names:
            if shape_name in self.heads:
                out = x
                for layer in self.heads[shape_name]:
                    out = (
                        layer(out, training=training)
                        if isinstance(layer, Dropout)
                        else layer(out)
                    )
                outputs.append(out)
        return tf.keras.layers.Concatenate()(outputs) if outputs else x


def build_task_model(
    nside,
    npix,
    npol,
    n_outputs,
    task_names,
    max_batch_size=128,
    pool_p=2,
    K=7,
    encoder_activation="gelu",
    head_activation="gelu",
    pool_type="AVG",
    dropout_rate=0.1,
    head_dropout=0.05,
):
    """Build encoder with task-specific heads of different capacities."""
    nside_factor = 2**pool_p
    depth = int(math.log(nside, nside_factor))
    level_nsides = [nside // (nside_factor**i) for i in range(depth + 1)]
    level_npixels = [12 * ns**2 for ns in level_nsides]
    channels = [npol] + [2 ** (i + 5) for i in range(depth + 1)]

    inputs = tf.keras.Input(shape=(npix, npol), name="lensed_maps")
    x = inputs

    for i in range(depth):
        x = EncoderBlock(
            nside=level_nsides[i],
            npix=level_npixels[i],
            fin=channels[i],
            fout=channels[i + 1],
            K=K,
            pool_p=pool_p,
            max_batch_size=max_batch_size,
            dropout_rate=dropout_rate,
            encoder_activation=encoder_activation,
            pool_type=pool_type,
        )(x)

    x = Flatten()(x)
    outputs = TaskSpecificHeads(
        n_outputs=n_outputs,
        task_names=task_names,
        activation=head_activation,
        dropout=head_dropout,
    )(x)

    return tf.keras.Model(inputs, outputs, name="task_conditioned_encoder")


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
    available_cpus = 16  # max(1, core.slurm.n_cpus or os.cpu_count() or 4)

    # Use in-memory caching to avoid concurrent file-based cache lockfile conflicts
    # when multiple Optuna trials run in parallel
    train, val, test = ds.split(
        train_size=float(fractions[0]),
        val_size=float(fractions[1]),
        test_size=float(fractions[2]),
        to_tf=True,
        batch_size=batch_size,
        duplicates=duplicates,
        buffer_size=32,
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
        K = trial.suggest_categorical("K", [5, 7, 9, 11, 13, 15, 32])

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

        # Head dropout for TaskSpecificHeads
        head_dropout = trial.suggest_float("head_dropout", 0.0, 0.3, step=0.05)

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

            model = build_task_model(
                nside=nside,
                npix=self.ctx.core.npix,
                npol=self.ctx.core.npols,
                n_outputs=len(self.ctx.core.shapes),
                task_names=self.ctx.core.shapes,
                max_batch_size=batch_size,
                pool_p=pool_p,
                K=K,
                encoder_activation=encoder_activation,
                head_activation=head_activation,
                pool_type=pool_type,
                dropout_rate=dropout_rate,
                head_dropout=head_dropout,
            )

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
    study_name = f"neo-task-n{ctx.core.nside}-{shapes_str}"

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
