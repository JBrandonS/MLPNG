"""
Neo Trainer: Encoder-based fnl prediction trainer with Weights & Biases logging.

This module provides a complete training pipeline for fnl prediction using
task-specific encoder architecture with support for multiple CMB bispectrum shapes
(local, equilateral, orthogonal).
"""

import os
import sys
import math
import logging
import argparse
import tracemalloc
import psutil
from datetime import datetime
from typing import Tuple, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

tf.get_logger().setLevel(logging.INFO)

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# Enable GPU memory growth to avoid allocating all GPU memory at once
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"GPU memory growth error: {e}")

from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")
from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import HealpyChebyshev, HealpyPool

import wandb
from wandb.integration.keras import WandbMetricsLogger

from mlpng import Core
from mlpng.utils import setup_logging, rmse_metrics
from mlpng.utils.dataloaders import KappaDataset

logger = logging.getLogger(__name__)


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder block with HEALPix Chebyshev convolution and pooling."""

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
        activation="gelu",
        pool_type="AVG",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nside = nside

        # Select initializer based on activation
        initializer = (
            tf.keras.initializers.HeNormal()
            if activation == "relu"
            else tf.keras.initializers.GlorotNormal()
        )

        layers = [
            HealpyChebyshev(
                K=K,
                Fout=fout,
                activation=activation,
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

    # Define head capacity for each shape: local (0), equilateral (1), orthogonal (2)
    SHAPE_TO_HEAD_INDEX = {
        "local": 0,
        "equilateral": 1,
        "orthogonal": 2,
    }

    HEAD_ARCHITECTURES = {
        0: [32, 32, 1],  # local: simple head
        1: [128, 64, 32, 1],  # equilateral: complex head
        2: [64, 64, 32, 1],  # orthogonal: medium head
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

        # Helper to build a head with given layer sizes
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

        # Build only heads for shapes present in task_names
        self.heads = {}
        for shape_name in self.task_names:
            if shape_name in self.SHAPE_TO_HEAD_INDEX:
                head_idx = self.SHAPE_TO_HEAD_INDEX[shape_name]
                head_arch = self.HEAD_ARCHITECTURES[head_idx]
                self.heads[shape_name] = build_head(
                    head_arch, f"head_{shape_name}", dropout, activation
                )
            else:
                logger.warning(f"Unknown shape: {shape_name}, skipping head creation")

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
    head_activation="relu",
    pool_type="AVG",
    dropout_rate=0.0,
    head_dropout=0.0,
):
    """Build encoder with task-specific heads of different capacities."""

    # Calculate encoder depth
    nside_factor = 2**pool_p
    depth = int(math.log(nside, nside_factor))
    level_nsides = [nside // (nside_factor**i) for i in range(depth + 1)]
    level_npixels = [12 * ns**2 for ns in level_nsides]
    channels = [npol] + [2 ** (i + 5) for i in range(depth + 1)]

    # Build encoder
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
            activation=encoder_activation,
            pool_type=pool_type,
        )(x)

    # Task-specific heads with different capacities
    x = Flatten()(x)
    outputs = TaskSpecificHeads(
        n_outputs=n_outputs,
        task_names=task_names,
        activation=head_activation,
        dropout=head_dropout,
    )(x)

    model = tf.keras.Model(inputs, outputs, name="task_conditioned_encoder")
    return model


def create_sigma_weighted_loss(sigma_all, weight_scale=None):
    """Create sigma-weighted MSE loss for inverse-variance weighting."""
    sigma_tensor = tf.constant(sigma_all, dtype=tf.float32)

    if weight_scale is None:
        weight_scale = [1.0] * len(sigma_all)
    weight_tensor = tf.constant(weight_scale, dtype=tf.float32)

    def sigma_weighted_mse(y_true, y_pred):
        squared_error = tf.square(y_true - y_pred)
        weights = weight_tensor / (sigma_tensor**2)
        weighted_loss = squared_error * weights
        return tf.reduce_mean(weighted_loss)

    return sigma_weighted_mse


class NeoTrainer:
    """Main trainer class for fnl prediction."""

    def __init__(
        self,
        core_args: list,
        data_fraction: float = 1.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 16,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            core_args: List of core arguments
            data_fraction: Fraction of data to use (0-1)
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            cache_dir: Directory for caching datasets
        """
        self.data_fraction = data_fraction
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        # Build Core CLI arguments
        self.core_args = core_args

        # Setup cache directory
        if cache_dir is None:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_dir = f"/lustre/smuexa01/client/users/stevensonb/tf_cache/"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Initialize core and data
        self.core = None
        self.sigma_all = None
        self.custom_loss = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.model = None
        self.history = None

        # Setup distributed training strategy for multi-GPU
        self.strategy = tf.distribute.MirroredStrategy()
        n_gpus = self.strategy.num_replicas_in_sync
        logger.info(f"Using MirroredStrategy with {n_gpus} device(s)")

    def log_memory(self, label: str = ""):
        """Log current memory usage at debug level."""
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_gb = mem_info.rss / 1024**3
            vms_gb = mem_info.vms / 1024**3
            logger.debug(
                f"[{label}] Memory - RSS: {rss_gb:.2f} GB, VMS: {vms_gb:.2f} GB"
            )
        except Exception as e:
            logger.debug(f"[{label}] Failed to get memory info: {e}")

    def get_dataset_memory_estimate(self, ds, name: str = "dataset"):
        """Estimate dataset memory usage by sampling first batch."""
        try:
            for x, y in ds.take(1):
                x_size_gb = x.numpy().nbytes / 1024**3
                y_size_gb = y.numpy().nbytes / 1024**3
                total_gb = x_size_gb + y_size_gb
                logger.debug(
                    f"{name} per-batch: X={x_size_gb:.3f} GB, Y={y_size_gb:.3f} GB, "
                    f"Total={total_gb:.3f} GB"
                )
        except Exception as e:
            logger.debug(f"Failed to estimate {name} memory: {e}")

    def setup(self):
        """Initialize core and datasets."""
        tracemalloc.start()
        self.log_memory("Before Core initialization")

        self.core = Core(self.core_args)
        self.shapes = self.core.shapes

        self.log_memory("After Core initialization")

        # Get sigma values
        self.sigma_all = np.array(self.core.get_likelihoods(True))
        self.custom_loss = create_sigma_weighted_loss(self.sigma_all)

        logger.info(f"Sigma values: {self.sigma_all}")

        # Calculate training parameters
        decay_steps = (
            self.core.total_sims * 0.8 * self.data_fraction * 25 // self.batch_size
        )
        logger.info(f"Decay steps: {decay_steps}")

        ds = KappaDataset.fromCore(
            self.core, x_output="lensed", y_output="fnl"  # , parallel_prefer="threads"
        )
        self.log_memory("After KappaDataset creation")

        # Create unique cache filename with nside and shapes to avoid lockfile conflicts
        shapes_str = (
            "-".join(self.core.shapes)
            if isinstance(self.core.shapes, list)
            else str(self.core.shapes)
        )
        cache_file = os.path.join(self.cache_dir, f"n{self.core.nside}_{shapes_str}")

        # Split dataset
        self.train_ds, self.val_ds, self.test_ds = ds.split(
            train_size=0.8 * self.data_fraction,
            val_size=0.1 * self.data_fraction,
            test_size=0.1 * self.data_fraction,
            to_tf=True,
            batch_size=self.batch_size,
            duplicates=[25, 10, 10],
            gen_batch_size=64,
            shuffle=False,
            cache_file=cache_file,
            buffer_size=1000,
        )
        self.log_memory("After dataset split")

        if cache_file is not None:
            logger.debug("Generating train cache")
            for i, _ in enumerate(self.train_ds):
                if i % 10 == 0:
                    logger.debug("  processed %d batches", i)
            self.log_memory("After train_ds caching")

            logger.debug("Generating val cache")
            for _ in self.val_ds:
                pass
            self.log_memory("After val_ds caching")

            logger.debug("Generating test cache")
            for _ in self.test_ds:
                pass
            self.log_memory("After test_ds caching")

        self.get_dataset_memory_estimate(self.train_ds, "train_ds")
        self.get_dataset_memory_estimate(self.val_ds, "val_ds")
        self.get_dataset_memory_estimate(self.test_ds, "test_ds")

        return decay_steps

    def build_model(
        self,
        pool_p: int = 2,
        K: int = 7,
        encoder_activation: str = "gelu",
        head_activation=None,
        pool_type: str = "AVG",
        dropout_rate: float = 0.1,
        head_dropout: float = 0.1,
    ):
        """Build the model."""
        if head_activation is None:
            head_activation = LeakyReLU(0.3)

        self.log_memory("Before model building")
        with self.strategy.scope():
            self.model = build_task_model(
                nside=self.core.nside,
                npix=self.core.npix,
                npol=self.core.npols,
                n_outputs=len(self.core.shapes),
                task_names=self.core.shapes,
                max_batch_size=self.batch_size,
                pool_p=pool_p,
                K=K,
                encoder_activation=encoder_activation,
                head_activation=head_activation,
                pool_type=pool_type,
                dropout_rate=dropout_rate,
                head_dropout=head_dropout,
            )

        self.log_memory("After model building")

    def train(
        self,
        initial_lr: float = 1e-5,
        decay_rate: float = 0.98,
        decay_steps: float = 1000,
        weight_decay: float = 1e-5,
        use_cosine_decay: bool = False,
        use_wandb: bool = True,
    ) -> Tuple:
        if use_wandb:
            wandb.init(
                project="mlpng",
                entity="mlpng",
                config={
                    # Data parameters
                    "nside": self.core.nside,
                    "npix": self.core.npix,
                    "npols": self.core.npols,
                    "shapes": self.core.shapes,
                    "n_outputs": len(self.core.shapes),
                    "nsims": self.core.total_sims,
                    "fnl_min": self.core.fnl_min,
                    "fnl_max": self.core.fnl_max,
                    "sigma_values": self.sigma_all.tolist(),
                    "data_fraction": self.data_fraction,
                    "batch_size": self.batch_size,
                    "decay_steps": decay_steps,
                    "max_epochs": self.max_epochs,
                    "patience": self.patience,
                    "use_cosine_decay": use_cosine_decay,
                    "initial_lr": initial_lr,
                    "decay_rate": decay_rate,
                    "weight_decay": weight_decay,
                    "loss": "mse",  # make sure to update this
                },
                tags=[f"nside-{self.core.nside}", "task", *self.core.shapes],
            )

        # Compile model within strategy scope for distributed training
        with self.strategy.scope():
            # Setup learning rate schedule
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

            optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

            self.model.compile(
                optimizer=optimizer,
                loss="mse",  # self.custom_loss, # update wandb.init loss above on change
                metrics=rmse_metrics(self.core.shapes),
            )

        # Setup callbacks
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ),
        ]

        if use_wandb:
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))

        self.model.summary()

        self.log_memory("Before model.fit()")
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        logger.info(
            f"Training complete! Best val_loss: {min(self.history.history['val_loss']):.6f}"
        )

        if use_wandb:
            wandb.finish()

        return self.model, self.history

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        test_predictions = self.model.predict(self.test_ds, verbose=0)

        # Collect ground truth
        test_truth = []
        for _, y in self.test_ds:
            test_truth.append(y.numpy())
        test_truth = np.concatenate(test_truth, axis=0)
        return test_predictions, test_truth

    def plot_training_history(self, save_dir: Optional[str] = None) -> str:
        if save_dir is None:
            save_dir = self.cache_dir

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.history.history["loss"], label="Train", linewidth=2)
        axes[0].plot(self.history.history["val_loss"], label="Val", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Training History")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale("log")

        # Per-shape RMSE
        for shape in self.core.shapes:
            val_key = f"val_rmse_{shape}"
            if val_key in self.history.history:
                axes[1].plot(
                    self.history.history[val_key],
                    label=shape,
                    marker="o",
                    markersize=3,
                    alpha=0.7,
                )

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Val RMSE per Shape")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = self.core.get_plot_file("loss", subdir="trainer")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training history plot: {filepath}")
        plt.close()

        return filepath

    def _plot_evaluation_base(
        self,
        predictions: np.ndarray,
        truth: np.ndarray,
        sigma_all: np.ndarray,
        plot_title: str,
        save_filename: str,
        xlim_range: Optional[Tuple[float, float]] = None,
        shape_masks: Optional[dict] = None,
        sample_counts: Optional[dict] = None,
    ) -> str:
        """
        Base plotting function for evaluation plots (test set and restricted range).

        Args:
            predictions: Model predictions shape (n_samples, n_shapes)
            truth: Ground truth fnl values shape (n_samples, n_shapes)
            sigma_all: Sigma values for each shape
            plot_title: Title for the figure
            save_filename: Base filename for saving
            xlim_range: Optional (xmin, xmax) for scatter plot x-axis
            shape_masks: Optional dict mapping shape names to boolean masks
            sample_counts: Optional dict mapping shape names to sample counts to display

        Returns:
            Path to saved figure.
        """
        nshapes = len(self.core.shapes)

        fig, axes = plt.subplots(2, nshapes, figsize=(5 * nshapes, 10))

        # Ensure axes is always 2D for consistent indexing across all cases
        if nshapes == 1:
            # plt.subplots(2, 1) returns shape (2,), reshape to (2, 1)
            axes = axes.reshape(2, 1)

        for shape_idx, shape_name in enumerate(self.core.shapes):
            # Apply mask if provided, otherwise use all data
            if shape_masks is not None and shape_name in shape_masks:
                mask = shape_masks[shape_name]
                truth_shape = truth[mask, shape_idx]
                pred_shape = predictions[mask, shape_idx]
                sample_count = mask.sum()
            else:
                truth_shape = truth[:, shape_idx]
                pred_shape = predictions[:, shape_idx]
                sample_count = len(truth_shape)

            error = pred_shape - truth_shape

            sigma = sigma_all[shape_idx] if len(sigma_all) > shape_idx else sigma_all[0]
            corr = (
                np.corrcoef(truth_shape, pred_shape)[0, 1]
                if len(truth_shape) > 1
                else np.nan
            )
            rmse = np.sqrt(np.mean(error**2))

            # Scatter plot
            if xlim_range:
                line = np.array(xlim_range)
                axes[0, shape_idx].set_xlim(xlim_range[0] - 1, xlim_range[1] + 1)
                axes[0, shape_idx].set_ylim(
                    xlim_range[0] - sigma - 1, xlim_range[1] + sigma + 1
                )
            else:
                line = np.array([np.nanmin(truth_shape), np.nanmax(truth_shape)])

            axes[0, shape_idx].scatter(truth_shape, pred_shape, alpha=0.4, s=8)
            axes[0, shape_idx].plot(line, line, "r--", label="Perfect", linewidth=2)
            axes[0, shape_idx].plot(line, line + sigma, "g--", alpha=0.5)
            axes[0, shape_idx].plot(line, line - sigma, "g--", alpha=0.5)
            axes[0, shape_idx].set_xlabel("True fnl")
            axes[0, shape_idx].set_ylabel("Predicted fnl")

            # Build title with sample count if available
            title_str = (
                f"{shape_name}\nCorr: {corr:.3f}, RMSE: {rmse:.2f}, σ: {sigma:.2f}"
            )
            if sample_count is not None:
                title_str = f"{shape_name} (N={sample_count})\nCorr: {corr:.3f}, RMSE: {rmse:.2f}, σ: {sigma:.2f}"

            axes[0, shape_idx].set_title(title_str)
            axes[0, shape_idx].grid(True, alpha=0.3)

            # Error histogram with Gaussian curve
            counts, bins, patches = axes[1, shape_idx].hist(
                error,
                bins=40 if shape_masks is None else 30,
                alpha=0.7,
                edgecolor="black",
            )

            # Add Gaussian distribution curve
            x = np.linspace(error.min(), error.max(), 100)
            gaussian = stats.norm.pdf(x, loc=0, scale=sigma)
            # Scale gaussian to match histogram
            gaussian_scaled = gaussian * len(error) * (bins[1] - bins[0])
            axes[1, shape_idx].plot(
                x, gaussian_scaled, color="purple", linewidth=2, label="Gaussian(σ)"
            )

            axes[1, shape_idx].axvline(
                sigma, color="g", linestyle="--", linewidth=2, label=f"+σ"
            )
            axes[1, shape_idx].axvline(
                -sigma, color="g", linestyle="--", linewidth=2, label=f"-σ"
            )
            axes[1, shape_idx].axvline(
                0, color="red", linestyle="-", linewidth=1, alpha=0.5
            )
            axes[1, shape_idx].set_xlabel("Prediction Error")
            axes[1, shape_idx].set_ylabel("Count")
            axes[1, shape_idx].set_title(f"Error Distribution (σ={sigma:.1f})")
            axes[1, shape_idx].grid(True, alpha=0.3, axis="y")
            axes[1, shape_idx].legend(fontsize=9)

        fig.suptitle(plot_title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = self.core.get_plot_file(save_filename, subdir="trainer")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved evaluation plot: {filepath}")
        plt.close()

        return filepath

    def plot_test_evaluation(
        self, predictions: np.ndarray, truth: np.ndarray, save_dir: Optional[str] = None
    ) -> str:
        """
        Plot per-shape test set evaluation (scatter plots and error histograms).

        Args:
            predictions: Model predictions shape (n_samples, n_shapes)
            truth: Ground truth fnl values shape (n_samples, n_shapes)
            save_dir: Directory to save plot. If None, uses core.get_plot_file().

        Returns:
            Path to saved figure.
        """
        sigma_all = self.core.get_likelihoods(True)

        return self._plot_evaluation_base(
            predictions=predictions,
            truth=truth,
            sigma_all=sigma_all,
            plot_title="Test Set Results - All Shapes",
            save_filename="evaluation",
        )

    def plot_restricted_range(
        self,
        predictions: np.ndarray,
        truth: np.ndarray,
        fnl_min: float = -100,
        fnl_max: float = 100,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Plot evaluation for restricted fnl range (per-shape filtering).

        Args:
            predictions: Model predictions shape (n_samples, n_shapes)
            truth: Ground truth fnl values shape (n_samples, n_shapes)
            fnl_min: Minimum fnl value for range
            fnl_max: Maximum fnl value for range
            save_dir: Directory to save plot. If None, uses core.get_plot_file().

        Returns:
            Path to saved figure.
        """
        sigma_all = self.core.get_likelihoods(True)

        # Create per-shape masks
        shape_masks = {}
        for shape_idx, shape_name in enumerate(self.core.shapes):
            shape_truth = truth[:, shape_idx]
            shape_masks[shape_name] = (shape_truth >= fnl_min) & (
                shape_truth <= fnl_max
            )

        return self._plot_evaluation_base(
            predictions=predictions,
            truth=truth,
            sigma_all=sigma_all,
            plot_title=f"Restricted Range [{fnl_min}, {fnl_max}] - Test Set Results (per-shape filtering)",
            save_filename="restricted",
            xlim_range=(fnl_min, fnl_max),
            shape_masks=shape_masks,
        )

    def get_summary_metrics(
        self, predictions: np.ndarray, truth: np.ndarray
    ) -> pd.DataFrame:
        """Compute summary metrics for each shape."""
        summary = []
        for shape_idx, shape_name in enumerate(self.core.shapes):
            truth_shape = truth[:, shape_idx]
            pred_shape = predictions[:, shape_idx]
            error = pred_shape - truth_shape

            sigma = self.sigma_all[shape_idx]

            summary.append(
                {
                    "Shape": shape_name,
                    "MAE": np.mean(np.abs(error)),
                    "RMSE": np.sqrt(np.mean(error**2)),
                    "RMSE/Sigma": np.sqrt(np.mean(error**2)) / sigma,
                    "Correlation": np.corrcoef(truth_shape, pred_shape)[0, 1],
                }
            )

        return pd.DataFrame(summary)


def main():
    """Command-line interface for the trainer using Core's argument handling."""
    setup_logging("mlpng.neo_trainer", level=logging.DEBUG)

    # Parse training-specific arguments from sys.argv
    # Core arguments (--config, --shapes, --nsims) are handled by Core
    data_fraction = 1.0
    batch_size = 128
    max_epochs = 100
    patience = 16
    cache_dir = None
    pool_p = 2
    K = 7
    dropout_rate = 0.1
    head_dropout = 0.00
    initial_lr = 5e-4
    decay_rate = 0.96
    weight_decay = 1e-6
    use_cosine_decay = True
    use_wandb = True

    # Simple argument parsing for neo_trainer specific options
    argv = sys.argv[1:]
    core_args = []

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "--data-fraction" and i + 1 < len(argv):
            data_fraction = float(argv[i + 1])
            i += 2
        elif arg == "--batch-size" and i + 1 < len(argv):
            batch_size = int(argv[i + 1])
            i += 2
        elif arg == "--max-epochs" and i + 1 < len(argv):
            max_epochs = int(argv[i + 1])
            i += 2
        elif arg == "--patience" and i + 1 < len(argv):
            patience = int(argv[i + 1])
            i += 2
        elif arg == "--cache-dir" and i + 1 < len(argv):
            cache_dir = argv[i + 1]
            i += 2
        elif arg == "--pool-p" and i + 1 < len(argv):
            pool_p = int(argv[i + 1])
            i += 2
        elif arg == "--K" and i + 1 < len(argv):
            K = int(argv[i + 1])
            i += 2
        elif arg == "--dropout" and i + 1 < len(argv):
            dropout_rate = float(argv[i + 1])
            i += 2
        elif arg == "--head-dropout" and i + 1 < len(argv):
            head_dropout = float(argv[i + 1])
            i += 2
        elif arg == "--lr" and i + 1 < len(argv):
            initial_lr = float(argv[i + 1])
            i += 2
        elif arg == "--decay-rate" and i + 1 < len(argv):
            decay_rate = float(argv[i + 1])
            i += 2
        elif arg == "--weight-decay" and i + 1 < len(argv):
            weight_decay = float(argv[i + 1])
            i += 2
        elif arg == "--use-cosine-decay":
            use_cosine_decay = True
            i += 1
        elif arg == "--no-wandb":
            use_wandb = False
            i += 1
        else:
            # Pass Core-related arguments to Core
            core_args.append(arg)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                core_args.append(argv[i + 1])
                i += 2
            else:
                i += 1

    logger.debug(f"Core arguments: {core_args}")

    # Create trainer - pass core_args as a single argument for Core initialization
    trainer = NeoTrainer(
        core_args=core_args,
        data_fraction=data_fraction,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        cache_dir=cache_dir,
    )

    decay_steps = trainer.setup()

    trainer.build_model(
        pool_p=pool_p,
        K=K,
        dropout_rate=dropout_rate,
        head_dropout=head_dropout,
    )

    trainer.train(
        initial_lr=initial_lr,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        weight_decay=weight_decay,
        use_cosine_decay=use_cosine_decay,
        use_wandb=use_wandb,
    )

    predictions, truth = trainer.evaluate()
    summary_df = trainer.get_summary_metrics(predictions, truth)

    print("\nTest Set Performance Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Generate plots
    trainer.plot_training_history()
    trainer.plot_test_evaluation(predictions, truth)
    trainer.plot_restricted_range(predictions, truth, fnl_min=-100, fnl_max=100)


if __name__ == "__main__":
    main()
