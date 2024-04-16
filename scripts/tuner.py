import logging
import os
import sys

# fix a issues with how these files are run in the script
# I probably just need to get better with python paths
# os.chdir("/users/stevensonb/Research/MLPNG/scripts")
sys.path.append("/users/stevensonb/Research/MLPNG/scripts")
sys.path.append("/users/stevensonb/Research/MLPNG/scripts/utils")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from utils import setup_logging

logger = setup_logging(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("WarmupLearningRate").setLevel(logging.ERROR)

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from trainer import alm_model
from .utils import Config, log_source, setup_logging
from .utils.tf.callbacks import WarmupLearningRate
from .utils.tf.dataloaders import *


def model_builder(hp):
    regularizers = ["", "l1", "l2", "l1_l2"]
    activations = ["", "tanh", "relu", "sigmoid", "swish"]
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    lr_schedule = WarmupLearningRate(
        warmup_learning_rate=hp.Float(
            "warmup_lr", min_value=1e-8, max_value=1e-3, sampling="log"
        ),
        warmup_steps=hp.Int(
            "warmup_steps", min_value=1e2, max_value=1e6, sampling="log"
        ),
        warmup_scale=1.5,
        warmup_scale_steps=10,
        warmed_learning_rate="auto",
        decay_steps=hp.Int("decay_steps", min_value=1e2, max_value=1e6, sampling="log"),
        decay_rate=0.95,
        staircase=True,
    )

    model = alm_model(
        Input(data_loader.shape),
        depth=hp.Choice("depth", values=[1, 2, 4, 6, 8, 16]),
        mha_num_heads=hp.Choice("mha_num_heads", values=[1, 2, 4, 8, 16, 32]),
        mha_initializer=hp.Choice(
            "mha_initializer",
            values=["glorot_uniform", "truncated_normal", "he_uniform"],
        ),
        mha_kernel_regularizer=hp.Choice("mha_kernel_regularizer", values=regularizers),
        mha_dropout=hp.Choice("mha_dropout", values=dropouts),
        div_key_dims=hp.Boolean("div_key_dims"),
        ff_hidden=hp.Choice("ff_hidden", values=[128, 256, 512, 1024, 2048]),
        ff_activation=hp.Choice("ff_activation", values=activations),
        ff_kernel_regularizer=hp.Choice("ff_kernel_regularizer", values=regularizers),
        ff_dropout=hp.Choice("ff_dropout", values=dropouts),
        final_hidden_1=hp.Choice("final_hidden_1", values=[256, 512, 1024, 2048]),
        final_hidden_2=hp.Choice("final_hidden_2", values=[128, 256, 512, 1024]),
        final_hidden_3=hp.Choice("final_hidden_3", values=[16, 32, 64, 128, 256, 512]),
        final_dropout=hp.Choice("final_dropout", values=dropouts),
        final_activation_1=hp.Choice("final_activation_1", values=activations),
        final_activation_2=hp.Choice("final_activation_2", values=activations),
        final_activation_3=hp.Choice("final_activation_3", values=activations),
        final_kernel_regularizer=hp.Choice(
            "final_kernel_regularizer", values=regularizers
        ),
    )

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.mse)
    return model


if __name__ == "__main__":
    s = Config()

    data_loader = AlmLoaderV2(s.alm_file, shuffle=True, batch_size=1, cache=True)
    train, test, val = data_loader.get_split(0.8, 0.1, 0.1)

    gpus = tf.config.list_physical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy([f"/gpu:{i}" for i in range(len(gpus))])

    # could maybe try the BayesianOptimization
    # tuner = kt.BayesianOptimization(
    #     model_builder,
    #     objective="val_loss",
    #     max_trials=100,
    #     directory="data/tuning",
    #     project_name="attention_2lm_bayesian_tiny",
    #     distribution_strategy=strategy,
    # )

    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        # max_epochs=100,
        # factor=10,
        hyperband_iterations=3,
        executions_per_trial=5,
        directory="data/tuning",
        project_name="attention_2lm_v2",
        distribution_strategy=strategy,
    )
    tuner.search_space_summary(True)

    tuner.search(
        train,
        epochs=5,
        validation_data=val,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
    )

    tuner.results_summary()
