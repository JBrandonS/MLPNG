import os
import sys
import math
import logging
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
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
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


class DeepEncoderBlock(tf.keras.layers.Layer):
    """Deep encoder block with double HEALPix Chebyshev convolution and pooling.

    Two graph convolutions per block before pooling, inspired by the double-conv
    pattern in U-Net. This gives the network more capacity to learn features at
    each resolution level before downsampling.
    """

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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nside = nside

        # Select initializer based on activation
        initializer = (
            tf.keras.initializers.HeNormal()
            if encoder_activation == "relu"
            else tf.keras.initializers.GlorotNormal()
        )

        # Double convolution: fin → fout → fout, then dropout, then pool
        layers = [
            HealpyChebyshev(
                K=K,
                Fout=fout,
                activation=encoder_activation,
                use_bn=True,
                use_bias=True,
                initializer=initializer,
            ),
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

    # Define head capacity for each shape: local (0), equilateral (1), orthogonal (2)
    SHAPE_TO_HEAD_INDEX = {
        "local": 0,
        "equilateral": 1,
        "orthogonal": 2,
    }

    # Larger heads for nside=128 deeper encoder
    HEAD_ARCHITECTURES = {
        0: [32, 32, 1],  # local (strongest signal)
        1: [64, 64, 32, 32, 1],  # equilateral (weaker signal, needs more capacity)
        2: [64, 32, 32, 1],  # orthogonal (weaker signal, needs more capacity)
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


def build_deep_task_model(
    nside,
    npix,
    npol,
    n_outputs,
    task_names,
    max_batch_size=64,
    pool_p=1,
    K_schedule=None,
    channels=None,
    encoder_activation="gelu",
    head_activation="gelu",
    pool_type="AVG",
    dropout_rate=0.1,
    head_dropout=0.05,
):
    """Build deep encoder with double-conv blocks, progressive K, and task-specific heads.

    Uses pool_p=1 (halving nside each level) for maximum depth.
    For nside=128 this gives 7 encoder blocks (14 graph convolutions total).

    Architecture per block:
        HealpyChebyshev(K, fin→fout) → HealpyChebyshev(K, fout→fout) → Dropout → HealpyPool

    Args:
        nside: HEALPix nside parameter
        npix: Number of pixels (12 * nside^2)
        npol: Number of polarization channels
        n_outputs: Number of output values
        task_names: List of shape names for task-specific heads
        max_batch_size: Maximum batch size for HealpyGCNN
        pool_p: Pooling parameter (1 = halve nside each level)
        K_schedule: List of K values per level. If None, uses progressive schedule.
        channels: List of channel sizes [fin, fout_0, fout_1, ...]. If None, auto-computed.
        encoder_activation: Activation function for encoder blocks
        head_activation: Activation function for task heads
        pool_type: Pooling type ("AVG" or "AVG")
        dropout_rate: Dropout rate in encoder blocks
        head_dropout: Dropout rate in task heads
    """
    nside_factor = 2**pool_p
    depth = int(math.log(nside, nside_factor))
    level_nsides = [nside // (nside_factor**i) for i in range(depth + 1)]
    level_npixels = [12 * ns**2 for ns in level_nsides]

    # Default channel schedule: gradual increase, capped at 256
    if channels is None:
        channels = [npol] + [2 ** (i + 4) for i in range(depth)]

    # Default K schedule: progressive increase with depth
    if K_schedule is None:
        K_schedule = [5 for i in range(depth)]

    logger.debug(f"Deep encoder architecture (depth={depth}):")
    logger.debug(f"  Level nsides:  {level_nsides}")
    logger.debug(f"  Level npixels: {level_npixels}")
    logger.debug(f"  Channels:      {channels}")
    logger.debug(f"  K schedule:    {K_schedule}")
    logger.debug(
        f"  Flatten size:  {level_npixels[-1]} x {channels[-1]} = {level_npixels[-1] * channels[-1]}"
    )

    inputs = tf.keras.Input(shape=(npix, npol), name="unlensed_maps")
    x = inputs

    for i in range(depth):
        x = DeepEncoderBlock(
            nside=level_nsides[i],
            npix=level_npixels[i],
            fin=channels[i],
            fout=channels[i + 1],
            K=K_schedule[i],
            pool_p=pool_p,
            max_batch_size=max_batch_size,
            dropout_rate=dropout_rate,
            encoder_activation=encoder_activation,
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

    model = tf.keras.Model(inputs, outputs, name="deep_task_encoder")
    return model


def create_sigma_weighted_loss(sigma_all):
    """Create sigma-weighted MSE loss for inverse-variance weighting."""
    sigma_tensor = tf.constant(sigma_all, dtype=tf.float32)

    def sigma_weighted_mse(y_true, y_pred):
        squared_error = tf.square(y_true - y_pred)
        weights = 1.0 / (sigma_tensor**2)
        weighted_loss = squared_error * weights
        return tf.reduce_mean(weighted_loss)

    return sigma_weighted_mse


class NeoTrainer:
    """Main trainer class for deep fnl prediction."""

    def __init__(
        self,
        core_args: list,
        data_fraction: float = 1.0,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            core_args: List of core arguments
            data_fraction: Fraction of data to use (0-1)
            batch_size: Batch size for training (default 64 for nside=128)
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
            cache_dir = f"/lustre/smuexa01/client/users/stevensonb/tf_cache/unlensed-4/"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Initialize core and data
        # self.core = None
        # self.sigma_all = None
        # self.custom_loss = None
        # self.train_ds = None
        # self.val_ds = None
        # self.test_ds = None
        # self.model = None
        # self.history = None

        # Setup distributed training strategy for multi-GPU
        self.strategy = tf.distribute.MirroredStrategy()
        n_gpus = self.strategy.num_replicas_in_sync
        logger.debug(f"Using MirroredStrategy with {n_gpus} device(s)")

    def setup(self):
        """Initialize core and datasets."""
        self.core = Core(self.core_args)
        self.shapes = self.core.shapes

        # Get sigma values
        self.sigma_all = np.array(self.core.get_likelihoods(True))
        self.custom_loss = create_sigma_weighted_loss(self.sigma_all)

        logger.debug(f"Sigma values: {self.sigma_all}")

        # Calculate training parameters
        decay_steps = (
            self.core.total_sims * 0.8 * 250 * self.data_fraction // self.batch_size
        )
        logger.debug(f"Decay steps: {decay_steps}")

        ds = KappaDataset.fromCore(self.core, x_output="unlensed", y_output="fnl")

        # Create unique cache filename with nside and shapes to avoid lockfile conflicts
        cache_file = os.path.join(
            self.cache_dir,
            f"n{self.core.nside}_unlensed_{self.core.shapes_str}_d{'_'.join(map(str, [250, 100, 20]))}",
        )

        # Split dataset
        gen_bs = 8 if self.core.nside >= 128 else 32
        self.train_ds, self.val_ds, self.test_ds = ds.split(
            train_size=0.8 * self.data_fraction,
            val_size=0.1 * self.data_fraction,
            test_size=0.1 * self.data_fraction,
            to_tf=True,
            batch_size=self.batch_size,
            duplicates=[250, 100, 20],
            gen_batch_size=gen_bs,
            cache_file=cache_file,
        )

        if cache_file is not None:
            logger.debug("Generating train cache")
            for _ in self.train_ds:
                pass

            logger.debug("Generating val cache")
            for _ in self.val_ds:
                pass

            logger.debug("Generating test cache")
            for _ in self.test_ds:
                pass

        return decay_steps

    def build_model(
        self,
        pool_p: int = 1,
        K_schedule: Optional[list] = None,
        channels: Optional[list] = None,
        encoder_activation: str = "gelu",
        head_activation="gelu",
        pool_type: str = "AVG",
        dropout_rate: float = 0.1,
        head_dropout: float = 0.05,
    ):
        """Build the deep model.

        Args:
            pool_p: Pooling parameter (1 = halve nside each level, 7 levels for nside=128)
            K_schedule: List of K values per level. If None, uses progressive schedule.
            channels: List of channel sizes. If None, auto-computed.
            encoder_activation: Activation function for encoder blocks
            head_activation: Activation function for task heads
            pool_type: Pooling type ("AVG" or "AVG")
            dropout_rate: Dropout rate in encoder blocks
            head_dropout: Dropout rate in task heads
        """
        if head_activation is None:
            head_activation = LeakyReLU(0.3)

        with self.strategy.scope():
            self.model = build_deep_task_model(
                nside=self.core.nside,
                npix=self.core.npix,
                npol=self.core.npols,
                n_outputs=len(self.core.shapes),
                task_names=self.core.shapes,
                max_batch_size=self.batch_size,
                pool_p=pool_p,
                K_schedule=K_schedule,
                channels=channels,
                encoder_activation=encoder_activation,
                head_activation=head_activation,
                pool_type=pool_type,
                dropout_rate=dropout_rate,
                head_dropout=head_dropout,
            )

    def train(
        self,
        initial_lr: float = 1e-3,
        decay_rate: float = 0.98,
        decay_steps: float = 1000,
        weight_decay: float = 1e-7,
        use_cosine_decay: bool = True,
        use_wandb: bool = True,
        K_schedule: Optional[list] = None,
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
                    # Deep model parameters
                    "model_type": "deep_task",
                    "K_schedule": K_schedule,
                    "head_architectures": TaskSpecificHeads.HEAD_ARCHITECTURES,
                    # Optimizer parameters
                    "use_cosine_decay": use_cosine_decay,
                    "initial_lr": initial_lr,
                    "decay_rate": decay_rate,
                    "weight_decay": weight_decay,
                    "loss": "mse",  # make sure to update this
                },
                tags=[
                    f"nside-{self.core.nside}",
                    "trainer",
                    "unlensed",
                    *self.core.shapes,
                ],
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
                loss="mse",  # update wandb.init loss above on change
                metrics=rmse_metrics(self.core.shapes),
            )

        # Setup callbacks with unique checkpoint path based on settings
        checkpoints_dir = os.path.join(self.core.dirs["model"], "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Generate unique checkpoint name from key training settings
        checkpoint_name = f"n{self.core.nside}_unlensed_{self.core.shapes_str}.h5"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        logger.info(f"Checkpoint path: {checkpoint_path}")

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=0,
            ),
        ]

        if use_wandb:
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))

        self.model.summary()

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

        return self.model, self.history

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        # Collect ground truth
        test_truth = np.concatenate([y.numpy() for _, y in self.test_ds], axis=0)

        # Predict — MirroredStrategy may pad the last batch, so truncate to match
        test_predictions = self.model.predict(self.test_ds, verbose=0)

        n = min(len(test_predictions), len(test_truth))
        test_predictions = test_predictions[:n]
        test_truth = test_truth[:n]

        logger.info(
            f"Evaluate: predictions {test_predictions.shape}, truth {test_truth.shape}"
        )
        return test_predictions, test_truth

    def plot_training_history(self):
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
        logger.debug(f"Saved training history plot: {filepath}")
        plt.close()

    def _plot_evaluation_base(
        self,
        predictions: np.ndarray,
        truth: np.ndarray,
        sigma_all: np.ndarray,
        # plot_title: str,
        save_filename: str,
        xlim_range: Optional[Tuple[float, float]] = None,
        shape_masks: Optional[dict] = None,
    ):
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
            # corr = (
            #     np.corrcoef(truth_shape, pred_shape)[0, 1]
            #     if len(truth_shape) > 1
            #     else np.nan
            # )
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

            axes[0, shape_idx].scatter(truth_shape, pred_shape, s=8)
            axes[0, shape_idx].plot(line, line, "r--", label="Perfect", linewidth=2)
            axes[0, shape_idx].plot(line, line + sigma, "g--")
            axes[0, shape_idx].plot(line, line - sigma, "g--")
            axes[0, shape_idx].set_xlabel(r"True $f_\mathrm{fnl}$")
            axes[0, shape_idx].set_ylabel(r"Predicted $f_\mathrm{fnl}$")
            axes[0, shape_idx].set_title(
                f"{shape_name} (N={sample_count}, RMSE: {rmse:.2f})"
            )
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
                x,
                gaussian_scaled,
                color="purple",
                linewidth=2,
                label="Gaussian",
                alpha=0.5,
            )

            axes[1, shape_idx].axvline(
                sigma,
                color="g",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label=f"σ({sigma:.1f})",
            )
            axes[1, shape_idx].axvline(
                -sigma, color="g", linestyle="--", linewidth=2, alpha=0.5
            )
            axes[1, shape_idx].axvline(
                0, color="red", linestyle="-", linewidth=1, alpha=0.5
            )
            axes[1, shape_idx].set_xlabel("Residuals")
            axes[1, shape_idx].set_ylabel("Frequency")
            # axes[1, shape_idx].set_title(f"Error Distribution (σ={sigma:.1f})")
            axes[1, shape_idx].grid(True, alpha=0.3, axis="y")
            axes[1, shape_idx].legend(fontsize=9)

        # fig.suptitle(plot_title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = self.core.get_plot_file(save_filename, subdir="trainer")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.debug(f"Saved evaluation plot: {filepath}")
        plt.close()

    def plot_test_evaluation(
        self, predictions: np.ndarray, truth: np.ndarray, save_dir: Optional[str] = None
    ):
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

        self._plot_evaluation_base(
            predictions=predictions,
            truth=truth,
            sigma_all=sigma_all,
            # plot_title="Test Set Results - All Shapes",
            save_filename="evaluation",
        )

    def plot_restricted_range(
        self,
        predictions: np.ndarray,
        truth: np.ndarray,
        fnl_min: float = -100,
        fnl_max: float = 100,
    ):
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

        self._plot_evaluation_base(
            predictions=predictions,
            truth=truth,
            sigma_all=sigma_all,
            # plot_title=f"Restricted Range [{fnl_min}, {fnl_max}] - Test Set Results (per-shape filtering)",
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

    def get_restricted_summary(self, predictions, truth, fnl_min=-100, fnl_max=100):
        """Compute summary metrics for restricted fnl range (per-shape filtering)."""
        restricted_summary = []
        for shape_idx, shape_name in enumerate(self.core.shapes):
            mask = (truth[:, shape_idx] >= fnl_min) & (truth[:, shape_idx] <= fnl_max)
            truth_shape = truth[mask, shape_idx]
            pred_shape = predictions[mask, shape_idx]
            error = pred_shape - truth_shape

            sigma = self.sigma_all[shape_idx]

            restricted_summary.append(
                {
                    "Shape": shape_name,
                    "N": mask.sum(),
                    "MAE": np.mean(np.abs(error)),
                    "RMSE": np.sqrt(np.mean(error**2)),
                    "RMSE/Sigma": np.sqrt(np.mean(error**2)) / sigma,
                    "Correlation": (
                        np.corrcoef(truth_shape, pred_shape)[0, 1]
                        if len(truth_shape) > 1
                        else np.nan
                    ),
                    "Sigma": sigma,
                }
            )

        return pd.DataFrame(restricted_summary)

    def get_comparison_metrics(
        self, predictions, truth, fnl_min=-100, fnl_max=100
    ) -> pd.DataFrame:
        """Compare full range vs restricted range metrics per shape."""
        comparison_data = []
        for shape_idx, shape_name in enumerate(self.core.shapes):
            # Full range metrics
            full_truth = truth[:, shape_idx]
            full_pred = predictions[:, shape_idx]
            full_error = full_pred - full_truth
            sigma = self.sigma_all[shape_idx]

            # Restricted range metrics (per-shape filtering)
            mask = (full_truth >= fnl_min) & (full_truth <= fnl_max)
            rest_truth = truth[mask, shape_idx]
            rest_pred = predictions[mask, shape_idx]
            rest_error = rest_pred - rest_truth

            comparison_data.append(
                {
                    "Shape": shape_name,
                    "Full N": len(full_truth),
                    "Full RMSE": np.sqrt(np.mean(full_error**2)),
                    "Full RMSE/\u03c3": np.sqrt(np.mean(full_error**2)) / sigma,
                    f"[{fnl_min},{fnl_max}] N": mask.sum(),
                    f"[{fnl_min},{fnl_max}] RMSE": np.sqrt(np.mean(rest_error**2)),
                    f"[{fnl_min},{fnl_max}] RMSE/\u03c3": np.sqrt(
                        np.mean(rest_error**2)
                    )
                    / sigma,
                    "\u03c3": sigma,
                }
            )

        return pd.DataFrame(comparison_data)


def main():
    """Command-line interface for the deep trainer.

    All arguments are passed directly to Core (settings file, --shapes, --nsims, etc.).
    Training hyperparameters are hardcoded below for nside=128 deep architecture.
    """
    setup_logging("mlpng.trainer_unlensed", level=logging.DEBUG)

    # Training hyperparameters for deep model (nside=128)
    batch_size = 128
    max_epochs = 50
    patience = 30
    pool_p = 2  # Halve nside each level → 7 levels for nside=128
    K_schedule = None  # [3, 5, 7, 7, 7, 5, 3]
    dropout_rate = 0.1
    head_dropout = 0.0
    initial_lr = 1e-3
    decay_rate = 0.98
    weight_decay = 1e-7
    use_cosine_decay = False

    # All CLI args go to Core (settings file, --shapes, --nsims, --wandb, etc.)
    trainer = NeoTrainer(
        core_args=sys.argv[1:],
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
    )

    decay_steps = trainer.setup()
    use_wandb = trainer.core.use_wandb

    trainer.build_model(
        pool_p=pool_p,
        K_schedule=K_schedule,
        dropout_rate=dropout_rate,
        head_dropout=head_dropout,
    )

    trainer.train(
        initial_lr=initial_lr,
        decay_rate=decay_rate,
        decay_steps=decay_steps * 5,
        weight_decay=weight_decay,
        use_cosine_decay=use_cosine_decay,
        use_wandb=use_wandb,
        K_schedule=K_schedule,
    )

    predictions, truth = trainer.evaluate()
    summary_df = trainer.get_summary_metrics(predictions, truth)
    restricted_df = trainer.get_restricted_summary(
        predictions, truth, fnl_min=-100, fnl_max=100
    )
    comparison_df = trainer.get_comparison_metrics(
        predictions, truth, fnl_min=-100, fnl_max=100
    )

    print("\nTest Set Performance Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    print("\nRestricted Range [-100, 100] Performance Summary (per-shape filtering)")
    print("=" * 100)
    print(restricted_df.to_string(index=False))
    print("=" * 100)

    print("\nComparison: Full Range vs Restricted Range (per-shape filtering)")
    print("=" * 120)
    print(comparison_df.to_string(index=False))
    print("=" * 120)

    # Generate plots
    trainer.plot_training_history()
    trainer.plot_test_evaluation(predictions, truth)
    trainer.plot_restricted_range(predictions, truth, fnl_min=-100, fnl_max=100)

    # Log final metrics to W&B and finish run
    if use_wandb:
        final_metrics = {
            "best_val_loss": min(trainer.history.history["val_loss"]),
            "final_epoch": len(trainer.history.history["loss"]),
        }

        for _, row in summary_df.iterrows():
            shape = row["Shape"]
            final_metrics[f"test/{shape}_mae"] = row["MAE"]
            final_metrics[f"test/{shape}_rmse"] = row["RMSE"]
            final_metrics[f"test/{shape}_rmse_sigma"] = row["RMSE/Sigma"]
            final_metrics[f"test/{shape}_correlation"] = row["Correlation"]

        for _, row in restricted_df.iterrows():
            shape = row["Shape"]
            final_metrics[f"test_restricted/{shape}_mae"] = row["MAE"]
            final_metrics[f"test_restricted/{shape}_rmse"] = row["RMSE"]
            final_metrics[f"test_restricted/{shape}_rmse_sigma"] = row["RMSE/Sigma"]
            final_metrics[f"test_restricted/{shape}_correlation"] = row["Correlation"]

        final_metrics["restricted_fnl_min"] = -100
        final_metrics["restricted_fnl_max"] = 100

        wandb.log(final_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
