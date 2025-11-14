"""
Optuna-based hyperparameter tuning script for U-Net model training.

This script uses Optuna to optimize hyperparameters for the U-Net model
that performs phi map reconstruction from lensed CMB maps.
"""

import os
import sys
import math
import logging
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Add
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import optuna
from optuna.integration import TFKerasPruningCallback

from deepsphere import HealpyGCNN
from deepsphere.healpy_layers import (
    HealpyChebyshev,
    HealpyPool,
    HealpyPseudoConv_Transpose,
)

from mlpng import Core
from mlpng.utils import setup_logging, make_trainer_plots
from mlpng.utils.dataloaders import PhiMapDataset
from mlpng.utils.callbacks import RMSELoss, rmse_metrics

tf.get_logger().setLevel(logging.ERROR)
logger = setup_logging("mlpng.optuna_trainer", level=logging.DEBUG)


class ResidualHealpyUNet:
    """
    Residual U-Net architecture for HEALPix maps with configurable hyperparameters.
    """

    def __init__(
        self,
        input_shape,
        activation="relu",
        max_batch_size=32,
        dropout_rate=0.1,
        use_bn=True,
        use_bias=True,
        k_values=None,
        base_channels_multiplier=1.0,
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
        self.k_values = k_values
        self.base_channels_multiplier = base_channels_multiplier

    def get_model(self):
        from mlpng.trainer import EncoderBlock, DecoderBlock

        depth = int(math.log2(self.nside))

        # Apply multiplier to base channels
        base_channels = [self.npol] + [
            int(2 ** (i + 4) * self.base_channels_multiplier) for i in range(depth)
        ]
        level_npixels = [self.npix // (4**i) for i in range(depth + 1)]
        level_nsides = [hp.npix2nside(npix) for npix in level_npixels]

        # Use provided K values or defaults
        if self.k_values is None:
            Ks = [1 + (2 * (1 + i // 3)) for i in range(depth + 1)]
        else:
            Ks = self.k_values

        logger.debug(f"Using Ks: {Ks}")
        logger.debug(f"Using base_channels: {base_channels}")

        inputs = tf.keras.Input(shape=self.input_shape[1:])
        x = inputs

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

        # Final block
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

        # Output
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


def objective(trial, core, splits, duplicates, cache_file, max_epochs, strategy, shapes):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        core: Core configuration object
        splits: Train/val/test split ratios
        duplicates: Data duplication factors
        cache_file: Path to cache file
        max_epochs: Maximum training epochs
        strategy: TensorFlow distribution strategy
        shapes: Shape configurations

    Returns:
        float: Validation loss to minimize
    """
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    initial_lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    k = trial.suggest_int("k", 1, 7, step=2)  # Chebyshev polynomial order
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "elu"])
    use_bn = trial.suggest_categorical("use_bn", [True, False])
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    base_channels_multiplier = trial.suggest_float("base_channels_multiplier", 0.5, 2.0)
    use_amsgrad = trial.suggest_categorical("use_amsgrad", [True, False])
    use_ema = trial.suggest_categorical("use_ema", [True, False])

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Trial {trial.number} - Testing hyperparameters:")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  learning_rate: {initial_lr:.6f}")
    logger.info(f"  dropout_rate: {dropout_rate:.3f}")
    logger.info(f"  k: {k}")
    logger.info(f"  activation: {activation}")
    logger.info(f"  use_bn: {use_bn}")
    logger.info(f"  weight_decay: {weight_decay:.6f}")
    logger.info(f"  base_channels_multiplier: {base_channels_multiplier:.2f}")
    logger.info(f"  use_amsgrad: {use_amsgrad}")
    logger.info(f"  use_ema: {use_ema}")
    logger.info(f"{'=' * 80}\n")

    # Create dataset
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

    # Calculate steps per epoch
    epoch_steps = math.ceil(core.total_sims * splits[0] * duplicates[0] // batch_size)
    decay_steps = epoch_steps * 3
    logger.debug(f"Steps per epoch: {epoch_steps}")

    # Build model with trial hyperparameters
    with strategy.scope():
        learning_rate = ExponentialDecay(initial_lr, decay_steps, 0.96, staircase=True)

        # Create K values list based on suggested k
        depth = int(math.log2(hp.npix2nside(core.npix)))
        k_values = [k] * (depth + 1)

        u_net = ResidualHealpyUNet(
            (None, core.npix, core.npols),
            activation=activation,
            max_batch_size=batch_size,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            k_values=k_values,
            base_channels_multiplier=base_channels_multiplier,
        ).get_model()

        u_net.compile(
            optimizer=AdamW(
                learning_rate,
                weight_decay=weight_decay,
                amsgrad=use_amsgrad,
                use_ema=use_ema,
            ),
            loss=RMSELoss(),
            metrics=rmse_metrics(shapes) + ["mse", "mae"],
        )

    # Callbacks
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]

    # Create caches
    logger.debug("Creating caches")
    for _ in train:
        pass
    for _ in val:
        pass
    logger.debug("Caches created")

    # Train model
    history = u_net.fit(
        train,
        epochs=max_epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=0,
    )

    # Return best validation loss
    best_val_loss = min(history.history["val_loss"])
    logger.info(f"Trial {trial.number} - Best validation loss: {best_val_loss:.6f}")

    return best_val_loss


def run_optuna_study(
    core,
    n_trials=50,
    study_name=None,
    storage=None,
    max_epochs=100,
):
    """
    Run Optuna hyperparameter optimization study.

    Args:
        core: Core configuration object
        n_trials: Number of trials to run
        study_name: Name of the study (for storage)
        storage: Optuna storage URL (e.g., sqlite:///optuna.db)
        max_epochs: Maximum epochs per trial

    Returns:
        optuna.Study: Completed study object
    """
    # Configuration
    data_fraction = 0.1
    splits = np.array([0.8, 0.1, 0.1]) * data_fraction
    duplicates = [25, 10, 2]

    strategy = tf.distribute.MirroredStrategy()

    date_time = tf.timestamp().numpy().astype(int)
    if study_name is None:
        study_name = f"unet-optuna-{date_time}"

    cache_file = f"{core.name}/unet-optuna-{core.name}"

    logger.info(f"Starting Optuna study: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Max epochs per trial: {max_epochs}")

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            core,
            splits,
            duplicates,
            cache_file,
            max_epochs,
            strategy,
            core.shapes,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_value:.6f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")

    # Save results
    save_dir = f"{core.dirs['model']}/{core.name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save study as dataframe
    df = study.trials_dataframe()
    df.to_csv(f"{save_dir}/optuna_study_{study_name}.csv", index=False)
    logger.info(f"Study results saved to: {save_dir}/optuna_study_{study_name}.csv")

    # Plot optimization history if matplotlib is available
    try:
        plot_dir = f"{core.dirs['plot']}/{core.name}/optuna"
        os.makedirs(plot_dir, exist_ok=True)

        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{plot_dir}/optimization_history_{study_name}.html")

        # Plot parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{plot_dir}/param_importances_{study_name}.html")

        # Plot parallel coordinate
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{plot_dir}/parallel_coordinate_{study_name}.html")

        logger.info(f"Optimization plots saved to: {plot_dir}")
    except Exception as e:
        logger.warning(f"Could not save plots: {e}")

    return study


def main():
    """Main entry point for the Optuna trainer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for U-Net training"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials to run",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the study (for storage and logging)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)",
    )

    args = parser.parse_args()

    # Initialize core
    core = Core()

    logger.info("=" * 80)
    logger.info("MLPNG Optuna Hyperparameter Optimization")
    logger.info("=" * 80)
    logger.info(f"Core configuration: {core.name}")
    logger.info(f"Number of pixels: {core.npix}")
    logger.info(f"Nside: {core.nside}")
    logger.info(f"Polarizations: {core.npols}")
    logger.info("=" * 80 + "\n")

    # Run study
    study = run_optuna_study(
        core=core,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        max_epochs=args.max_epochs,
    )

    logger.info("Optuna optimization complete!")

    return 0


if __name__ == "__main__":
    print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "unknown"))
    print("Python executable:", sys.executable)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Optuna version: {optuna.__version__}")

    sys.exit(main())
