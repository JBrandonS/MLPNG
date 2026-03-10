"""
Phi Scale Analysis: Train ML models at each phi_scale and compare R_GL with analytical values.

Loads pre-computed analytical R_GL data from ratio_ploter.py output, trains a
DeepEncoderBlock + TaskSpecificHeads model at each phi_scale, evaluates RMSE,
and computes R_GL^ML = sigma_Fisher / RMSE_ML. Generates comparison plots.

Uses the deep encoder architecture from neo_trainer_2.py: double-conv blocks,
progressive K schedule, and MAX pooling.

Models are saved to disk so incomplete runs can be resumed.

Usage:
    python -m mlpng.phi_scale_analysis settings/n256.json --pols T --shapes local --lensing
"""

import gc
import os
import sys
import math
import time
import logging
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"GPU memory growth error: {e}")

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for batch jobs
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay
from tensorflow.keras.models import load_model

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import HealpyChebyshev, HealpyPool

from mlpng import Core
from mlpng.utils import setup_logging, rmse_metrics, save_data
from mlpng.utils.dataloaders import KappaDataset

logger = logging.getLogger(__name__)

# Phi scales to train ML models at (subset of ratio_ploter PHI_SCALES)
# PHI_SCALES_ML = [1, 5, 10, 50, 100, 200, 500]
PHI_SCALES_ML = [1, 10, 100, 500]


# =============================================================================
# Model Architecture (from neo_trainer_2.py)
# =============================================================================


@tf.keras.saving.register_keras_serializable(package="phi_scale")
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nside = nside
        self._npix = npix
        self._fin = fin
        self._fout = fout
        self._K = K
        self._pool_p = pool_p
        self._max_batch_size = max_batch_size
        self._dropout_rate = dropout_rate
        self._encoder_activation = encoder_activation
        self._pool_type = pool_type

        # Select initializer based on activation
        initializer = (
            tf.keras.initializers.HeNormal()
            if encoder_activation == "relu"
            else tf.keras.initializers.GlorotNormal()
        )

        # Double convolution: fin -> fout -> fout, then dropout, then pool
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nside": self.nside,
                "npix": self._npix,
                "fin": self._fin,
                "fout": self._fout,
                "K": self._K,
                "pool_p": self._pool_p,
                "max_batch_size": self._max_batch_size,
                "dropout_rate": self._dropout_rate,
                "encoder_activation": self._encoder_activation,
                "pool_type": self._pool_type,
            }
        )
        return config

    def call(self, x, training=False):
        return self.body(x, training=training)


@tf.keras.saving.register_keras_serializable(package="phi_scale")
class TaskSpecificHeads(tf.keras.layers.Layer):
    """Shape-dependent heads with increasing capacity for local, equilateral, orthogonal."""

    SHAPE_TO_HEAD_INDEX = {"local": 0, "equilateral": 1, "orthogonal": 2}
    # Larger heads for deeper encoder
    HEAD_ARCHITECTURES = {
        0: [64, 32, 32, 1],  # local (strongest signal)
        1: [128, 64, 64, 1],  # equilateral (weaker signal, needs more capacity)
        2: [64, 64, 32, 32, 1],  # orthogonal (weaker signal, needs more capacity)
    }

    def __init__(self, n_outputs, task_names, activation="relu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.task_names = (
            task_names if isinstance(task_names, (list, tuple)) else [task_names]
        )
        self.activation = activation
        self._dropout = dropout

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
        return (
            tf.keras.layers.Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_outputs": self.n_outputs,
                "task_names": list(self.task_names),
                "activation": self.activation,
                "dropout": self._dropout,
            }
        )
        return config


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
        HealpyChebyshev(K, fin->fout) -> HealpyChebyshev(K, fout->fout) -> Dropout -> HealpyPool

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
        pool_type: Pooling type ("AVG" or "MAX")
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
        K_schedule = [min(5 + i, 12) for i in range(depth)]

    logger.info(f"Deep encoder architecture (depth={depth}):")
    logger.info(f"  Level nsides:  {level_nsides}")
    logger.info(f"  Level npixels: {level_npixels}")
    logger.info(f"  Channels:      {channels}")
    logger.info(f"  K schedule:    {K_schedule}")
    logger.info(
        f"  Flatten size:  {level_npixels[-1]} x {channels[-1]} = {level_npixels[-1] * channels[-1]}"
    )

    inputs = tf.keras.Input(shape=(npix, npol), name="lensed_maps")
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


# =============================================================================
# Main analysis class
# =============================================================================


class PhiScaleAnalysis:
    """Train ML models per phi_scale and compare R_GL with analytical values."""

    def __init__(self, core_args=None):
        # Training hyperparameters
        self.data_fraction = 1.0
        self.duplicates = [25, 10, 2]
        self.max_epochs = 100
        self.patience = 32
        self.batch_size = 128

        # Model hyperparameters (matching neo_trainer_2 deep encoder)
        self.pool_p = 1
        self.K_schedule = None  # progressive: [min(5+i, 12) for i in range(depth)]
        self.channels = None  # auto: [npol] + [2**(i+4) for i in range(depth)]
        self.encoder_activation = "gelu"
        self.head_activation = "gelu"
        self.pool_type = "AVG"
        self.dropout_rate = 0.1
        self.head_dropout = 0.05

        # Optimizer settings
        self.use_cosine_decay = True
        self.initial_lr = 1e-4
        self.decay_rate = 0.98
        self.weight_decay = 1e-5

        # Initialize Core
        if core_args is None:
            core_args = sys.argv[1:]
        self.core = Core(core_args)

        self.decay_steps = int(
            self.core.total_sims * 0.8 * self.data_fraction // self.batch_size
        )

        # Paths
        self.data_file = self.core.file
        self.ratio_data_file = self.data_file.replace("10000", "1000")
        self.model_dir = os.path.join(self.core.dirs["model"], "phi_scale_analysis_2")
        os.makedirs(self.model_dir, exist_ok=True)

        self.cache_dir = "/lustre/smuexa01/client/users/stevensonb/tf_cache/"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Fisher sigma values
        self.sigma_fisher_unlensed = self.core.get_likelihoods(False)[0]
        self.sigma_fisher_lensed = self.core.get_likelihoods(True)[0]

        # Distributed strategy
        self.strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using {self.strategy.num_replicas_in_sync} device(s)")

        logger.info(f"Data file (ML):    {self.data_file}")
        logger.info(f"Data file (ratio): {self.ratio_data_file}")
        logger.info(f"Model dir:         {self.model_dir}")
        logger.info(
            f"nside={self.core.nside}, lmax={self.core.lmax}, "
            f"npix={self.core.npix}, npol={self.core.npols}"
        )
        logger.info(f"shapes={self.core.shapes}")
        logger.info(f"σ_Fisher (unlensed) = {self.sigma_fisher_unlensed:.4f}")
        logger.info(f"σ_Fisher (lensed)   = {self.sigma_fisher_lensed:.4f}")

    def load_analytical_rgl(self):
        """Load analytical R_GL data from ratio_ploter output."""
        logger.info(f"Loading analytical R_GL from {self.ratio_data_file}...")
        try:
            with h5py.File(self.ratio_data_file, "r") as f:
                rp = f["ratio_plot"]
                self.phi_scales_analytical = rp["phi_scales"][:]
                self.R_gl_analytical = rp["R_gl"][:]
                self.N2_0 = float(rp["N2_0"][0])
                self.N2_lens = float(rp["N2_lens"][0])
                self.delta_N2_analytical = rp["delta_N2"][:]
                self.lmax_rp = int(rp["lmax"][0])

            logger.info(f"Loaded analytical R_GL data (lmax={self.lmax_rp})")
            logger.info(f"  phi_scales: {self.phi_scales_analytical}")
            logger.info(f"  R_GL:       {np.round(self.R_gl_analytical, 4)}")
            logger.info(f"  N2_0 = {self.N2_0:.6e}, N2_lens = {self.N2_lens:.6e}")

        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(
                f"Analytical R_GL data not found in {self.ratio_data_file}.\n"
                "Run `python -m mlpng.ratio_ploter settings/n256.json "
                "--pols T --shapes local --lensing` first."
            ) from e

    def create_dataset(self, phi_scale, cache_suffix=""):
        """Create train/val/test datasets for a given phi_scale."""
        ds = KappaDataset.fromCore(
            self.core,
            phi_scale=float(phi_scale),
            x_output="lensed",
            y_output="fnl",
        )
        train_ds, val_ds, test_ds = ds.split(
            train_size=0.8 * self.data_fraction,
            val_size=0.1 * self.data_fraction,
            test_size=0.1 * self.data_fraction,
            to_tf=True,
            batch_size=self.batch_size,
            duplicates=self.duplicates,
            gen_batch_size=64,
            cache_dir=self.cache_dir,
            cache_file=f"phi-analysis-3-n{self.core.nside}{cache_suffix}",
        )
        return train_ds, val_ds, test_ds

    def _build_model(self):
        """Build a new deep task model."""
        return build_deep_task_model(
            nside=self.core.nside,
            npix=self.core.npix,
            npol=self.core.npols,
            n_outputs=len(self.core.shapes),
            task_names=self.core.shapes,
            max_batch_size=self.batch_size,
            pool_p=self.pool_p,
            K_schedule=self.K_schedule,
            channels=self.channels,
            encoder_activation=self.encoder_activation,
            head_activation=self.head_activation,
            pool_type=self.pool_type,
            dropout_rate=self.dropout_rate,
            head_dropout=self.head_dropout,
        )

    def train_model(self, model, train_ds, val_ds, model_name="model"):
        """Compile and train a model."""
        if self.use_cosine_decay:
            lr_schedule = CosineDecayRestarts(
                self.initial_lr,
                self.decay_steps,
                t_mul=2.0,
                m_mul=0.95,
                alpha=0.01,
            )
        else:
            lr_schedule = ExponentialDecay(
                self.initial_lr,
                self.decay_steps,
                self.decay_rate,
                staircase=True,
            )

        with self.strategy.scope():
            model.compile(
                optimizer=AdamW(
                    learning_rate=lr_schedule, weight_decay=self.weight_decay
                ),
                loss="mse",
                metrics=rmse_metrics(self.core.shapes),
            )

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            ),
        ]

        start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=2,
        )
        train_time = time.time() - start
        return history, train_time

    def evaluate_model(self, model, test_ds):
        """Evaluate model on test set."""
        preds = model.predict(test_ds, verbose=0)
        truth = np.concatenate([y.numpy() for _, y in test_ds])

        if len(self.core.shapes) == 1:
            truth = truth.ravel()
            preds = preds.ravel()

        rmse = np.sqrt(np.mean((preds - truth) ** 2))
        return preds, truth, rmse

    def load_or_train(self, model_path, train_ds, val_ds, model_name="model"):
        """Load existing model or train a new one. Supports resuming incomplete runs."""
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            with self.strategy.scope():
                model = load_model(
                    model_path, custom_objects={"EncoderBlock": DeepEncoderBlock}
                )
            gc.collect()
            tf.keras.backend.clear_session()
            return model, None, 0

        logger.info(f"Training new model: {model_name}")
        with self.strategy.scope():
            model = self._build_model()
        history, train_time = self.train_model(
            model, train_ds, val_ds, model_name=model_name
        )
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        gc.collect()
        return model, history, train_time

    def run(self, phi_scales=None):
        """Run the full phi_scale analysis pipeline."""
        if phi_scales is None:
            phi_scales = PHI_SCALES_ML

        # Step 1: Load analytical R_GL
        self.load_analytical_rgl()

        # Step 2: Plot analytical R_GL
        self._plot_analytical_rgl()

        # Step 3: Train and evaluate at each phi_scale
        results = {}
        for phi_scale in phi_scales:
            logger.info(f"{'='*60}")
            logger.info(f"Processing phi_scale = {phi_scale}")
            logger.info(f"{'='*60}")

            model_path = os.path.join(self.model_dir, f"task-phi{phi_scale}.keras")

            train_ds, val_ds, test_ds = self.create_dataset(
                phi_scale,
                cache_suffix=f"-phi{phi_scale}",
            )

            model, history, train_time = self.load_or_train(
                model_path,
                train_ds,
                val_ds,
                model_name=f"task-phi{phi_scale}",
            )

            preds, truth, rmse = self.evaluate_model(model, test_ds)
            R_gl_ml = self.sigma_fisher_unlensed / rmse

            results[phi_scale] = {
                "rmse": rmse,
                "R_gl_ml": R_gl_ml,
                "history": history,
                "train_time": train_time,
                "preds": preds,
                "truth": truth,
            }

            logger.info(f"  RMSE = {rmse:.4f}")
            logger.info(f"  R_GL_ML = σ_Fisher / RMSE = {R_gl_ml:.4f}")
            logger.info(f"  Train time = {train_time:.1f}s")

            del model
            gc.collect()
            tf.keras.backend.clear_session()

        logger.info("All phi_scales processed!")

        # Step 4: Save results
        self._save_results(results)

        # Step 5: Generate comparison plots
        self._plot_comparison(results)
        self._plot_evaluation(results)
        self._print_summary(results)

        logger.info("Done!")

    def _save_results(self, results):
        """Save ML results to HDF5."""
        phi_scales_ml = np.array(sorted(results.keys()), dtype=np.float64)
        rmses_ml = np.array(
            [results[ps]["rmse"] for ps in phi_scales_ml], dtype=np.float64
        )
        R_gl_ml = np.array(
            [results[ps]["R_gl_ml"] for ps in phi_scales_ml], dtype=np.float64
        )
        train_times = np.array(
            [results[ps]["train_time"] for ps in phi_scales_ml], dtype=np.float64
        )

        ml_data = {
            "phi_scale_analysis": {
                "phi_scales": phi_scales_ml,
                "rmse": rmses_ml,
                "R_gl_ml": R_gl_ml,
                "train_times": train_times,
                "sigma_fisher_unlensed": np.atleast_1d(self.sigma_fisher_unlensed),
                "sigma_fisher_lensed": np.atleast_1d(self.sigma_fisher_lensed),
            }
        }
        save_data(self.data_file, ml_data)
        logger.info(
            f"ML results saved to {self.data_file} under 'phi_scale_analysis' group"
        )

    def _get_plot_dir(self):
        """Get and create the plot output directory."""
        plot_dir = os.path.join(
            self.core.dirs["plot"],
            self.core.name,
            "phi_scale_analysis",
        )
        os.makedirs(plot_dir, exist_ok=True)
        return plot_dir

    def _plot_analytical_rgl(self):
        """Recreate the ratio_ploter R_GL vs phi_scale plot."""
        plot_dir = self._get_plot_dir()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            self.phi_scales_analytical,
            self.R_gl_analytical,
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=6,
            label=r"$R_{GL}$ (Analytical)",
        )
        ax.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=r"$R=1$ (no degradation)",
        )
        ax.set_xlabel(r"$\phi_{\mathrm{scale}}$", fontsize=14)
        ax.set_ylabel(r"$R_{GL} = (S/N)_{GL} / (S/N)_0$", fontsize=14)
        ax.set_title(
            rf"S/N Degradation vs Lensing Scale ($\ell_{{\max}}={self.core.lmax}$, nside={self.core.nside})",
            fontsize=14,
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        plt.tight_layout()

        plot_file = os.path.join(plot_dir, "R_gl_analytical.png")
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Analytical R_GL plot saved to {plot_file}")

    def _plot_comparison(self, results):
        """Overlay analytical R_GL with ML R_GL."""
        plot_dir = self._get_plot_dir()

        phi_scales_ml = np.array(sorted(results.keys()))
        R_gl_ml = np.array([results[ps]["R_gl_ml"] for ps in phi_scales_ml])

        fig, ax = plt.subplots(figsize=(12, 7))

        # Analytical curve
        ax.plot(
            self.phi_scales_analytical,
            self.R_gl_analytical,
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=5,
            label=r"Analytical (KSW): $R_{GL} = \sqrt{N^2_0 / (N^2_{\mathrm{lens}} + \delta N^2)}$",
        )

        # ML points
        ax.plot(
            phi_scales_ml,
            R_gl_ml,
            "s-",
            color="tab:red",
            linewidth=2,
            markersize=8,
            label=r"ML (TaskModel): $R_{GL}^{ML} = \sigma_{Fisher} / \mathrm{RMSE}_{ML}$",
        )

        # Reference lines
        ax.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=r"$R = 1$ (no degradation)",
        )

        R_fisher_lensed = self.sigma_fisher_unlensed / self.sigma_fisher_lensed
        ax.axhline(
            y=R_fisher_lensed,
            color="tab:green",
            linestyle=":",
            alpha=0.7,
            label=rf"Fisher lensing only: $R = {R_fisher_lensed:.3f}$",
        )

        ax.set_xlabel(r"$\phi_{\mathrm{scale}}$", fontsize=14)
        ax.set_ylabel(r"$R_{GL}$", fontsize=14)
        ax.set_title(
            rf"$R_{{GL}}$ Comparison: Analytical vs ML "
            rf"($\ell_{{\max}}={self.core.lmax}$, nside={self.core.nside}, local shape)",
            fontsize=14,
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        plt.tight_layout()

        plot_file = os.path.join(plot_dir, "R_gl_analytical_vs_ml.png")
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Comparison plot saved to {plot_file}")

    def _plot_evaluation(self, results):
        """Per-phi_scale scatter plots and error histograms."""
        plot_dir = self._get_plot_dir()
        phi_scales = sorted(results.keys())
        n_phis = len(phi_scales)
        ncols = min(4, n_phis)
        nrows = math.ceil(n_phis / ncols) * 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 2 and ncols == 1:
            axes = axes.reshape(nrows, ncols)

        for idx, phi_scale in enumerate(phi_scales):
            res = results[phi_scale]
            truth = res["truth"]
            preds = res["preds"]
            error = preds - truth
            rmse = res["rmse"]

            col = idx % ncols
            scatter_row = (idx // ncols) * 2
            hist_row = scatter_row + 1

            # Scatter plot
            ax_scatter = axes[scatter_row, col]
            ax_scatter.scatter(truth, preds, alpha=0.3, s=5, color="tab:blue")
            lims = [min(truth.min(), preds.min()), max(truth.max(), preds.max())]
            ax_scatter.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
            ax_scatter.fill_between(
                lims,
                [l - self.sigma_fisher_unlensed for l in lims],
                [l + self.sigma_fisher_unlensed for l in lims],
                alpha=0.1,
                color="red",
                label=rf"$\pm\sigma_{{Fisher}}$",
            )
            ax_scatter.set_xlabel(r"True $f_{NL}$")
            ax_scatter.set_ylabel(r"Predicted $f_{NL}$")
            ax_scatter.set_title(rf"$\phi_{{scale}}={phi_scale}$, RMSE={rmse:.2f}")
            ax_scatter.legend(fontsize=8)
            ax_scatter.set_aspect("equal")

            # Error histogram
            ax_hist = axes[hist_row, col]
            ax_hist.hist(
                error,
                bins=50,
                alpha=0.7,
                color="tab:orange",
                edgecolor="black",
                linewidth=0.5,
            )
            ax_hist.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax_hist.axvline(
                self.sigma_fisher_unlensed,
                color="r",
                linestyle=":",
                alpha=0.7,
                label=rf"$+\sigma_{{Fisher}}$",
            )
            ax_hist.axvline(
                -self.sigma_fisher_unlensed,
                color="r",
                linestyle=":",
                alpha=0.7,
                label=rf"$-\sigma_{{Fisher}}$",
            )
            ax_hist.set_xlabel(r"Error ($\hat{f}_{NL} - f_{NL}$)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(
                rf"$\phi_{{scale}}={phi_scale}$, mean err={np.mean(error):.2f}"
            )
            ax_hist.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_phis, ncols * (nrows // 2)):
            col = idx % ncols
            scatter_row = (idx // ncols) * 2
            hist_row = scatter_row + 1
            if scatter_row < nrows:
                axes[scatter_row, col].set_visible(False)
            if hist_row < nrows:
                axes[hist_row, col].set_visible(False)

        plt.tight_layout()
        plot_file = os.path.join(plot_dir, "per_phi_evaluation.png")
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Evaluation plots saved to {plot_file}")

    def _print_summary(self, results):
        """Print summary table."""
        R_gl_interp = interp1d(
            self.phi_scales_analytical,
            self.R_gl_analytical,
            kind="linear",
            fill_value="extrapolate",
        )

        header = (
            f"{'phi_scale':>10} | {'RMSE':>10} | {'σ_Fisher':>10} | "
            f"{'R_GL (Anal.)':>14} | {'R_GL (ML)':>12} | {'Train Time':>12}"
        )
        logger.info(header)
        logger.info("-" * 80)
        for phi_scale in sorted(results.keys()):
            res = results[phi_scale]
            R_gl_a = float(R_gl_interp(phi_scale))
            logger.info(
                f"{phi_scale:>10.0f} | {res['rmse']:>10.4f} | "
                f"{self.sigma_fisher_unlensed:>10.4f} | {R_gl_a:>14.4f} | "
                f"{res['R_gl_ml']:>12.4f} | {res['train_time']:>10.1f}s"
            )


if __name__ == "__main__":
    setup_logging(
        __name__,
        level=logging.DEBUG,
        scripts_level=logging.DEBUG,
        base_level=logging.ERROR,
    )

    analysis = PhiScaleAnalysis()
    analysis.run()
