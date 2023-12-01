import os
import sys
import h5py

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 1
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pandas as pd

plt.rcParams.update({"font.size": 13})

keras = tf.keras  # fixes issues with vsCode

from config import SimConfig
from dataloader import DataLoader
from isensee import isensee2017_model
from isensee_attn import isensee_attn

from keras import Input, Model
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Add, Dense, Flatten, RandomFlip, RandomRotation
from keras.metrics import KLDivergence, RootMeanSquaredError
from keras.optimizers import Adam
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

from sklearn.metrics import r2_score

import seaborn as sns

def load_data(data_file, keys, start_index=None, end_index=None, verbose=False):
    if isinstance(keys, str):
        keys = [keys]

    data = {}

    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        for key in keys:

            kv = hdf.get(key, None)
            if kv is None:
                raise ValueError(f"Key {key} not found in {data_file}")

            if start_index is not None and end_index is not None:
                data[key] = np.array(kv[start_index:end_index])  # type: ignore
            else:
                data[key] = np.array(kv[()])  # type: ignore

    return data

def safe_makedirs(dir, verbose=False):
    "Create a directory if it does not exist. Handles a race condition"
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            if verbose:
                print(f"Created directory {dir}")
        except FileExistsError:
            pass


def get_data(start, step, name=None):
    ret = d.skip(start).take(step)

    # This just fixes the logging output not knowing the dataset size
    ret = ret.apply(tf.data.experimental.assert_cardinality(step))

    ret = ret.cache()
    # ret = ret.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, name=name) # misaline in the data
    ret = ret.prefetch(tf.data.AUTOTUNE)
    return ret

def plot_preds(y_val, y_pred, name, fisher=None):
    df = pd.DataFrame({'True Labels': y_val.flatten(), 'Predicted Labels': y_pred.flatten()})

    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='True Labels', y='Predicted Labels')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--') 

    # plt.plot([min(y_val), max(y_val)], [mean, mean], color='green', linestyle='--')
    if fisher is not None:
        std_dev = np.sqrt(1/fisher)
        plt.plot([min(y_val), max(y_val)], [min(y_val) + std_dev, max(y_val) + std_dev], color='blue', linestyle='--')
        plt.plot([min(y_val), max(y_val)], [min(y_val) - std_dev, max(y_val) - std_dev], color='blue', linestyle='--')

    # Line for perfect fit
    r2 = r2_score(df['True Labels'], df['Predicted Labels'])
    plt.text(min(y_val), max(y_val), f'R^2 = {r2:.2f}', verticalalignment='top')

    plt.title(name)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{s.plot_dir}/{name}.png')


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings/settings.json"
    s = SimConfig(config_file)

    fisher = load_data(s.data_file_complete, ["fisher"], verbose=s.verbose)["fisher"]
    std_dev = np.sqrt(1/fisher)

    if s.debug or s.verbose:
        print(f"tf version: {tf.__version__}")

    batch_size = 64
    max_epochs = 500

    # load data as a generator so we do not need to have it all in memory
    d = tf.data.Dataset.from_generator(
        DataLoader(s.data_file_complete),
        output_signature=(
            tf.TensorSpec(shape=(s.nside, s.nside, 1), dtype=s.r_dtype),
            tf.TensorSpec(shape=(), dtype=s.r_dtype),
        ),
    )

    # fix a log warning
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    d = d.with_options(options)

    # Setup the data split as 80/10/10
    n = s.total_sims
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = int(n * 0.1)

    # loads in the datasets
    train_dataset = get_data(0, train_size, "train")
    val_dataset = get_data(train_size, val_size, "val")
    test_dataset = get_data(train_size + val_size, test_size, "test")
    
    X_train = []
    y_train = []
    for features, labels in train_dataset:
        # we normalize the data to be between 0 and 1
        min_val = np.min(features)
        max_val = np.max(features)
        features = (features - min_val) / (max_val - min_val)
        X_train.append(features)
        y_train.append(labels)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for features, labels in test_dataset:
        min_val = np.min(features)
        max_val = np.max(features)
        features = (features - min_val) / (max_val - min_val)
        X_test.append(features)
        y_test.append(labels)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_val = []
    y_val = []
    for features, labels in val_dataset:
        min_val = np.min(features)
        max_val = np.max(features)
        features = (features - min_val) / (max_val - min_val)
        X_val.append(features)
        y_val.append(labels)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    model_settings = {
        "depth": 5,
        "n_segmentation_levels": 3,
        "dropout_rate": 0.3,
        "loss_function": tf.keras.losses.mse,
        "initial_learning_rate": 0.01,
        "name": f"isensee-{s.base_name}"
    }

    wandb.init(
        project="mlpng",
        notes="isensee",
        tags=["isensee", "dev"],
        reinit=True,
        tensorboard=True,
        #    sync_tensorboard=True,
        config=s.settings | model_settings,
    )

    callbacks = [
        # EarlyStopping(
        #     monitor="val_loss",
        #     patience=10,
        #     verbose=1,
        #     restore_best_weights=True,
        #     start_from_epoch=50,
        # ),
        ModelCheckpoint(f'data/models/isensee-{s.base_name}'+'-{epoch:03d}.tf', monitor='val_loss', save_best_only=True, mode='auto'),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10
        ),
        WandbMetricsLogger(),
        # WandbModelCheckpoint(filepath=f"{s.model_dir}/wandb"),
        TensorBoard(log_dir=s.tb_dir),
    ]

    input_img = Input((s.nside, s.nside, 1), name="img")

    # setup the gpus, using mirrored to parallelize the training
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") # for debugging

    if False:
        with strategy.scope():
            isensee_model = isensee2017_model(input_img, **model_settings)

        if s.debug or s.verbose:
            isensee_model.summary()

        isensee_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        y_pred = isensee_model.predict(
            test_dataset,
            batch_size=batch_size,
            verbose="auto",
            callbacks=callbacks,
            use_multiprocessing=True,
        )

        # finish wandb so we get 2 models
        wandb.finish(0)

        plot_preds(y_truth, y_pred, model_settings["name"], fisher)

###############################################

    model_settings = {
        "depth": 5,
        "n_segmentation_levels": 3,
        "dropout_rate": 0.2,
        "loss_function": tf.keras.losses.mse,
        "initial_learning_rate": 0.001,
        "name": f"isensee_attn_{s.base_name}",
    }

    wandb.init(
        project="mlpng",
        notes="attention",
        tags=["isensee-attn", "dev"],
        reinit=True,
        config=s.settings | model_settings,
    )

    with strategy.scope():
        attn_model = isensee_attn(input_img, **model_settings)

    if s.debug or s.verbose:
        attn_model.summary()

    callbacks = [
        # EarlyStopping(
        #     monitor="val_loss",
        #     patience=10,
        #     verbose=1,
        #     restore_best_weights=True,
        #     start_from_epoch=50,
        # ),
        ModelCheckpoint(f'data/models/isensee_attn-{s.base_name}'+'-{epoch:03d}.tf', monitor='val_loss', save_best_only=True, mode='auto'),
        # ReduceLROnPlateau(
        #     monitor="val_loss", factor=0.5, patience=10
        # ),
        WandbMetricsLogger(),
        # WandbModelCheckpoint(filepath=f"{s.model_dir}/wandb"),
        TensorBoard(log_dir=s.tb_dir),
    ]

    attn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=4,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = attn_model.predict(
        test_dataset, batch_size=4, verbose="auto", callbacks=callbacks
    )
    
    plot_preds(y_test, y_pred, model_settings["name"], fisher)
