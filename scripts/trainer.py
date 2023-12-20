import os
import sys
import time

import h5py

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 1
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

plt.rcParams.update({"font.size": 13})

from tensorflow import keras

import seaborn as sns
from config import SimConfig
from dataloader import DataLoader
from tfdsdataloader import TFDSDataLoader
from isensee import isensee2017_model
from isensee_attn import isensee_attn
from isensee_joe import isensee2017_joe
from keras import Input, Model
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.layers import Add, Dense, Flatten, RandomFlip, RandomRotation
from keras.metrics import KLDivergence, RootMeanSquaredError
from keras.optimizers.legacy import Adam
from keras.regularizers import l2
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
import logging

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


def plot_preds(y_val, y_pred, name, fisher=None, scaled_variance=None):
    df = pd.DataFrame(
        {"True Labels": y_val.flatten(), "Predicted Labels": y_pred.flatten()}
    )

    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="True Labels", y="Predicted Labels")

    plt.plot(
        [min(y_val), max(y_val)], [min(y_val), max(y_val)], color="red", linestyle="--"
    )
    if fisher is not None:
        std_dev = np.sqrt(1 / fisher)
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) + std_dev, max(y_val) + std_dev],
            color="blue",
            linestyle="--",
            label="Fisher"
        )
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) - std_dev, max(y_val) - std_dev],
            color="blue",
            linestyle="--",
        )

    if scaled_variance is not None:
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) + scaled_variance, max(y_val) + scaled_variance],
            color="green",
            linestyle="--",
            label="Scaled Variance (1/$sqrt{f_{sky} f}$)})"
        )
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) - scaled_variance, max(y_val) - scaled_variance],
            color="green",
            linestyle="--",
        )

    # Line for perfect fit
    r2 = r2_score(df["True Labels"], df["Predicted Labels"])
    plt.text(min(y_val), max(y_val), f"R^2 = {r2:.2f}", verticalalignment="top")

    plt.title(name)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{s.plot_dir}/{name}.png")


def plot_history(attn_history, name, metrics=["loss"]):
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, figsize=(15, 6 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        axs[i].plot(attn_history.history[metric])
        axs[i].plot(attn_history.history[f"val_{metric}"])
        axs[i].set_title(f"Model {metric}")
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Epoch")
        axs[i].legend(["Train", "Validation"], loc="upper right")

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{s.plot_dir}/{name}-metrics.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S'
    )
    
    config_file = sys.argv[1]
    s = SimConfig(config_file)

    IMG_SHAPE = (s.nside, s.nside, 1)

    if s.debug or s.verbose:
        print(f"tf version: {tf.__version__}")

    batch_size = 1
    max_epochs = 1

    # load data as a generator so we do not need to have it all in memory
    # data_loader = DataLoader(
    #     s.data_file_complete, shuffle=True, seed=None, normalize=True
    # )
    # train_dataset, test_dataset, val_dataset = data_loader.get_split_tfdataset(
    #     0.8, 0.1, 0.1, batch_size=batch_size
    # )
    # print(data_loader)

    # lets try tfds, must be converted first
    tfds_filepath = s.data_file_complete.replace(".hdf5", ".tfds")
    data_loader = TFDSDataLoader(
        tfds_filepath, shuffle=True, seed=None, normalize=True
    )
    train_dataset, test_dataset, val_dataset = data_loader.get_split_tfdataset(
        0.008, 0.001, 0.001, batch_size=batch_size
    )

    # These get passed into the isensee_attn model, doing this here so we can save them into
    # wandb for later analysis
    timestamp = int(time.time())
    model_settings = {
        "depth": 5,
        "n_segmentation_levels": 3,
        "dropout_rate": 0.3,
        "loss_function": tf.keras.losses.mse,
        "initial_learning_rate": 0.001,
        "name": f"isensee_attn_{s.base_name}-{timestamp}",
        "n_base_filters": 8,
        "n_labels": 1,
        "interpolation": "nearest",
        "kernel_regularizer": l2(1e-8),
        "attn_heads": 1,
        "attn_key_dim": 4,
    }

    # Just gather some more info for the wandb run, helps later
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID") or 0,
        "start_time": timestamp,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "comment": "testing tfds",
    }

    wandb.init(
        project="mlpng",
        notes="attention",
        tags=["isensee-attn", "dev"],
        # reinit=True,
        config=s.settings | model_settings | extra_info,
    )

    lr_schedule = ExponentialDecay(
        initial_learning_rate=model_settings["initial_learning_rate"],
        decay_steps=10000,
        decay_rate=0.95,
        staircase=True,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=10,
        ),
        ModelCheckpoint(
            f"data/models/isensee_attn-{s.base_name}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
        WandbMetricsLogger(),
        # WandbModelCheckpoint(filepath=f"{s.model_dir}/wandb"),
        # TensorBoard(log_dir=s.tb_dir),
    ]

    metrics = ["mean_squared_error", "mean_absolute_error"]
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_img = Input(IMG_SHAPE, name="img")

        opt = Adam(
            learning_rate=lr_schedule,
            # learning_rate=model_settings["initial_learning_rate"],
        )

        # joe_model = isensee2017_joe(
        #     input_img,
        #     depth=5,
        #     n_segmentation_levels=3,
        #     dropout_rate=0.3,
        #     loss_function=tf.keras.losses.mse,
        #     initial_learning_rate=0.001,
        # )

        attn_model = isensee_attn(
            input_img,
            optimizer=opt,
            metrics=metrics,
            **model_settings,
        )

    if s.debug or s.verbose:
        attn_model.summary()

        print("Number of GPUs being used:", strategy.num_replicas_in_sync)
        print("Comment:", extra_info["comment"])

    # joe_history = joe_model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=max_epochs,
    #     callbacks=callbacks,
    #     verbose=2,
    # )

    attn_history = attn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    try:
        import keract # pip install keract for this to work

        first_batch = next(iter(test_dataset.take(1)))
        images, labels = first_batch
        activations = keract.get_activations(attn_model, images, auto_compile=True)

        # attn = activations.get("multi_head_attention_3")

        keract.display_activations(activations, 
                                   save=True, 
                                   directory=os.path.join(s.plot_dir, "activations", s.base_name), 
                                   data_format="channels_last")
    except Exception as e:
        import traceback
        print(f"Could not plot activations: {e}")
        traceback.print_exc()
        pass

    y_pred = attn_model.predict(test_dataset, callbacks=callbacks, verbose=2,)
    y_test = np.concatenate([y.numpy() for x, y in test_dataset])
    fisher = load_data(s.data_file_complete, ["fisher"], verbose=s.verbose)["fisher"]

    full_sky_degree = (np.pi / 180)**2
    f_sky = (s.settings["patch_side_deg"]) ** 2 / full_sky_degree
    scaled_variance = np.sqrt(1 / (f_sky * fisher))

    plot_preds(y_test, y_pred, model_settings["name"], fisher, scaled_variance)
    plot_history(
        attn_history,
        model_settings["name"],
        metrics=["loss", "mean_squared_error", "mean_absolute_error"],
    )
