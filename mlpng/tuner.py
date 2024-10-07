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

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from .models import AutoModel


def model_builder(hp):
    # regularizers = ["", "l1", "l2", "l1_l2"]
    # activations = ["", "tanh", "relu", "sigmoid", "swish"]
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    lr_schedule = ExponentialDecay(
        initial_learning_rate=hp.Float(
            "initial_lr", min_value=1e-8, max_value=1e-3, sampling="log"
        ),
        decay_steps=hp.Int("decay_steps", min_value=1e2, max_value=1e6, sampling="log"),
        decay_rate=hp.Float("decay_rate", min_value=0.1, max_value=0.99, step=0.01),
        staircase=hp.Boolean("staircase"),
    )

    model = AutoModel()
    model.init_dataset()
    model.make_model(
        depth=hp.Choice("depth", values=[1, 2, 4, 6, 8, 16]),
        dropout_rate=hp.Choice("dropout_rate", values=dropouts),
        ff_density=hp.Choice("ff_density", values=[64, 128, 256, 512, 1024, 2048]),
        mha_num_heads=hp.Choice("mha_num_heads", values=[1, 2, 4, 8, 16, 32]),
        mha_dropout=hp.Choice("mha_dropout", values=dropouts),
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.mse)
    return model.keras_model()


if __name__ == "__main__":
    model = AutoModel()

    dataset = model.init_dataset(
        shuffle=True,
        seed=None,
        batch_size=model.BATCH_SIZE,
        cache=True,
        shuffle_buffer=1000,
        normalize=True,
    )
    train, test, val = dataset.get_split(0.8, 0.1, 0.1)

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
