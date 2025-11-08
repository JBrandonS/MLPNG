"""Optuna-based hyperparameter search for the DeepSphere UNet."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

sys.path.append("/users/stevensonb/Research/tools/deepsphere-cosmo-tf2")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import optuna

import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay

from optuna.integration import TFKerasPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from mlpng import Core
from mlpng.unet_phi import get_model
from mlpng.utils import setup_logging
from mlpng.utils.callbacks import RMSELoss, rmse_metrics
from mlpng.utils.dataloaders import PhiMapDataset

logger = setup_logging("mlpng.unet_optuna", level=logging.INFO)


class UNetOptunaTrainer:
    def __init__(
        self,
        core: Core | None = None,
        n_trials: int = 500,
        study_name: str = "unet-optuna",
        storage: str | None = "sqlite:///data/tuner/unet.db",
        direction: str = "minimize",
        batch_size: int = 32,
        max_epochs: int = 60,
        patience: int = 8,
        duplicates: Tuple[int, int, int] = (25, 10, 2),
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        gen_batch_size: int = 8,
        buffer_size: int = 128,
        pruning: bool = True,
        seed: int = 42,
        output_dir: str | None = None,
        verbosity: int = 0,
    ):
        """Initialize the UNet Optuna trainer.

        Args:
            core: Core instance. If None, a new Core instance will be created.
            n_trials: Number of Optuna trials.
            study_name: Name of the Optuna study.
            storage: Optuna storage URL (e.g. sqlite:///unet-optuna.db).
            direction: Optuna optimization direction ('minimize' or 'maximize').
            batch_size: Training batch size.
            max_epochs: Maximum epochs per Optuna trial.
            patience: Early stopping patience in epochs.
            duplicates: Duplication factors for train, val, and test datasets.
            split: Dataset split fractions (must sum to 1.0).
            gen_batch_size: Batch size for generator datasets (data loader internal).
            buffer_size: Buffer size for dataset shuffling.
            pruning: Enable Optuna Hyperband pruning.
            seed: Random seed for reproducibility.
            output_dir: Directory to store the best model and study summary.
            verbosity: Keras training verbosity level (0=silent, 2=per epoch).
        """
        self.core = core if core is not None else Core()
        self.shapes = self.core.shapes
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbosity = verbosity
        self.pruning_enabled = pruning
        self.ds_split = np.array(split, dtype=float)
        self.duplicates = tuple(int(x) for x in duplicates)
        self.gen_batch_size = gen_batch_size
        self.buffer_size = buffer_size

        if not np.isclose(self.ds_split.sum(), 1.0):
            raise ValueError("Split fractions must sum to 1.0")

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.strategy = tf.distribute.MirroredStrategy()
        self.train_ds, self.val_ds, self.test_ds = self._prepare_datasets()
        self.input_shape = (None, self.core.npix, self.core.npols)
        self.steps_per_epoch = self._estimate_steps(
            self.ds_split[0], self.duplicates[0]
        )
        self.decay_steps = max(1, self.steps_per_epoch * 2)

        output_base = (
            Path(output_dir)
            if output_dir
            else Path(self.core.dirs["model"]) / self.core.name
        )
        output_base.mkdir(parents=True, exist_ok=True)
        self.best_model_path = output_base / f"{study_name}-best.keras"
        self.study_summary_path = output_base / f"{study_name}-summary.json"

        self.best_val_loss = math.inf
        self.best_trial_id: int | None = None
        self.best_params: Dict[str, Any] | None = None

    def _estimate_steps(self, frac: float, dups: int) -> int:
        part_size = self.core.total_sims // 3
        samples = math.ceil(part_size * frac * max(1, dups))
        return max(1, math.ceil(samples / self.batch_size))

    def _prepare_datasets(self):
        logger.info("Preparing datasets for UNet training")
        total_sims = self.core.total_sims
        part_size = total_sims // 3

        ds_unet = PhiMapDataset.fromCore(self.core, lensed=True)
        ds_unet.start_idx = 0
        ds_unet.end_idx = part_size

        train, val, test = ds_unet.split(
            train_size=self.ds_split[0],
            val_size=self.ds_split[1],
            test_size=self.ds_split[2],
            to_tf=True,
            batch_size=self.batch_size,
            duplicates=self.duplicates,
            gen_batch_size=self.gen_batch_size,
            buffer_size=self.buffer_size,
        )

        # Prefetch to overlap producer and consumer.
        autotune = tf.data.AUTOTUNE
        train = train.prefetch(autotune)
        val = val.prefetch(autotune)
        test = test.prefetch(autotune)

        return train, val, test

    def _build_schedule(
        self, trial: optuna.Trial
    ) -> tf.keras.optimizers.schedules.LearningRateSchedule | float:
        init_lr = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        schedule_type = trial.suggest_categorical(
            "lr_schedule", ["constant", "exponential", "cosine"]
        )

        if schedule_type == "exponential":
            decay_rate = trial.suggest_float("lr_decay", 0.80, 0.99)
            staircase = trial.suggest_categorical("lr_staircase", [True, False])
            return ExponentialDecay(init_lr, self.decay_steps, decay_rate, staircase)

        if schedule_type == "cosine":
            t_mul = trial.suggest_float("cosine_t_mul", 1.0, 3.0)
            m_mul = trial.suggest_float("cosine_m_mul", 0.5, 1.0)
            alpha = trial.suggest_float("cosine_alpha", 0.0, 0.2)
            return CosineDecayRestarts(
                init_lr, self.decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha
            )

        return init_lr

    def _model_kwargs(self, trial: optuna.Trial) -> Dict[str, Any]:
        # For unet_phi model, we can use either LeakyReLU with alpha or "selu"
        activation_type = trial.suggest_categorical(
            "activation_type", ["leaky_relu", "selu", "gelu", "relu"]
        )

        if activation_type == "leaky_relu":
            encoder_alpha = trial.suggest_float("encoder_alpha", 0.1, 0.4)
            decoder_alpha = trial.suggest_float("decoder_alpha", 0.1, 0.4)
            bottleneck_alpha = trial.suggest_float("bottleneck_alpha", 0.1, 0.4)
            encoder_activation = LeakyReLU(encoder_alpha)
            decoder_activation = LeakyReLU(decoder_alpha)
            bottleneck_activation = LeakyReLU(bottleneck_alpha)
        else:
            encoder_activation = activation_type
            decoder_activation = activation_type
            bottleneck_activation = activation_type

        return {
            "max_batch_size": self.batch_size,
            "use_double": trial.suggest_categorical("use_double", [True, False]),
            "pooling": trial.suggest_categorical("pooling", ["AVG", "MAX"]),
            "encoder_activation": encoder_activation,
            "encoder_k": trial.suggest_int("encoder_k", 1, 4),
            "encoder_dropout": trial.suggest_float(
                "encoder_dropout", 0.0, 0.3, step=0.1
            ),
            "bottleneck_layers": trial.suggest_int("bottleneck_layers", 1, 3),
            "bottleneck_activation": bottleneck_activation,
            "bottleneck_k": trial.suggest_int("bottleneck_k", 1, 4),
            "bottleneck_dropout": trial.suggest_float(
                "bottleneck_dropout", 0.0, 0.3, step=0.1
            ),
            "decoder_activation": decoder_activation,
            "decoder_k": trial.suggest_int("decoder_k", 1, 4),
            "decoder_dropout": trial.suggest_float(
                "decoder_dropout", 0.0, 0.3, step=0.1
            ),
            "final_k": trial.suggest_int("final_k", 1, 4),
            "final_bn": trial.suggest_categorical("final_bn", [True, False]),
            "final_activation": None,
        }

    def objective(self, trial: optuna.Trial) -> float:
        tf.keras.backend.clear_session()

        schedule = self._build_schedule(trial)
        model_kwargs = self._model_kwargs(trial)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ),
        ]
        if self.pruning_enabled:
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))

        with self.strategy.scope():
            model = get_model(self.input_shape, **model_kwargs)
            optimizer = AdamW(learning_rate=schedule, weight_decay=weight_decay)
            model.compile(
                optimizer=optimizer,
                loss=RMSELoss(),
                metrics=["mae", "mse", *rmse_metrics(self.shapes)],
            )

        history = model.fit(
            self.train_ds,
            epochs=self.max_epochs,
            validation_data=self.val_ds,
            callbacks=callbacks,
            verbose=self.verbosity,
        )

        val_metrics = model.evaluate(self.val_ds, return_dict=True, verbose=0)
        val_loss = float(val_metrics["loss"])

        trial.set_user_attr(
            "history", {k: [float(x) for x in v] for k, v in history.history.items()}
        )
        trial.set_user_attr(
            "val_metrics", {k: float(v) for k, v in val_metrics.items()}
        )

        if val_loss < self.best_val_loss:
            logger.info("New best trial %s with val_loss=%.5f", trial.number, val_loss)
            self.best_val_loss = val_loss
            self.best_trial_id = trial.number
            self.best_params = trial.params
            model.save(self.best_model_path, overwrite=True)

        # tf.keras.backend.clear_session()
        # gc.collect()
        return val_loss

    def finalize(self) -> Dict[str, Any]:
        if self.best_model_path.exists():
            logger.info("Loading best model from %s", self.best_model_path)
            best_model = tf.keras.models.load_model(self.best_model_path)
            test_metrics = best_model.evaluate(
                self.test_ds, return_dict=True, verbose=0
            )
        else:
            logger.warning(
                "Best model path %s not found; skipping test evaluation",
                self.best_model_path,
            )
            test_metrics = {}

        summary = {
            "best_trial": self.best_trial_id,
            "best_val_loss": self.best_val_loss,
            "best_params": self.best_params,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        }

        with self.study_summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        logger.info("Wrote study summary to %s", self.study_summary_path)
        return summary


def main(
    core: Core | None = None,
    n_trials: int = 500,
    study_name: str = "unet-optuna",
    storage: str | None = "sqlite:///data/tuner/unet.db",
    direction: str = "minimize",
    batch_size: int = 32,
    max_epochs: int = 60,
    patience: int = 8,
    duplicates: Tuple[int, int, int] = (25, 10, 2),
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    gen_batch_size: int = 8,
    buffer_size: int = 128,
    pruning: bool = True,
    seed: int = 42,
    output_dir: str | None = None,
    verbosity: int = 2,
) -> int:
    """Run Optuna hyperparameter optimization for UNet.

    Args:
        core: Core instance. If None, a new Core instance will be created.
        n_trials: Number of Optuna trials.
        study_name: Name of the Optuna study.
        storage: Optuna storage URL (e.g. sqlite:///unet-optuna.db).
        direction: Optuna optimization direction ('minimize' or 'maximize').
        batch_size: Training batch size.
        max_epochs: Maximum epochs per Optuna trial.
        patience: Early stopping patience in epochs.
        duplicates: Duplication factors for train, val, and test datasets.
        split: Dataset split fractions (must sum to 1.0).
        gen_batch_size: Batch size for generator datasets (data loader internal).
        buffer_size: Buffer size for dataset shuffling.
        pruning: Enable Optuna Hyperband pruning.
        seed: Random seed for reproducibility.
        output_dir: Directory to store the best model and study summary.
        verbosity: Keras training verbosity level (0=silent, 2=per epoch).

    Returns:
        Exit code (0 for success).
    """
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,
        multivariate=True,
    )

    pruner = (
        HyperbandPruner(
            min_resource=max(2, max_epochs // 6),
            max_resource=max_epochs,
            reduction_factor=3,
        )
        if pruning
        else None
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info("Sampler: %s", sampler.__class__.__name__)
    if pruner is None:
        logger.info("Pruning: disabled")
    else:
        min_res = getattr(pruner, "min_resource", None)
        if min_res is None:
            min_res = getattr(pruner, "_min_resource", "?")
        max_res = getattr(pruner, "max_resource", None)
        if max_res is None:
            max_res = getattr(pruner, "_max_resource", "?")
        logger.info(
            "Pruner: %s (min_resource=%s, max_resource=%s, reduction_factor=%s)",
            pruner.__class__.__name__,
            min_res,
            max_res,
            getattr(pruner, "reduction_factor", "?"),
        )

    trainer = UNetOptunaTrainer(
        core=core,
        n_trials=n_trials,
        study_name=study_name,
        storage=storage,
        direction=direction,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        duplicates=duplicates,
        split=split,
        gen_batch_size=gen_batch_size,
        buffer_size=buffer_size,
        pruning=pruning,
        seed=seed,
        output_dir=output_dir,
        verbosity=verbosity,
    )

    try:
        study.optimize(trainer.objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optuna optimization interrupted by user.")

    summary = trainer.finalize()

    logger.info("Best trial: %s", study.best_trial.number if study.best_trial else None)
    logger.info(
        "Best value: %.5f", study.best_value if study.best_trial else float("nan")
    )
    logger.info("Best params: %s", study.best_params if study.best_trial else {})
    logger.info("Summary: %s", summary)

    return 0


if __name__ == "__main__":
    logger.info("Conda environment: %s", os.environ.get("CONDA_DEFAULT_ENV", "unknown"))
    logger.info("Python executable: %s", sys.executable)
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("CUDA version: %s", tf.sysconfig.get_build_info().get("cuda_version"))
    logger.info("cuDNN version: %s", tf.sysconfig.get_build_info().get("cudnn_version"))

    sys.exit(main())
