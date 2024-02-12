import os
import sys
import time
import pprint
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.environ['CUDA_HOME']}"

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Add,
    Attention,
    MultiHeadAttention,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    Multiply,
    UpSampling2D,
    RandomFlip,
    RandomRotation,
    Dropout,
    LayerNormalization,
    Conv2DTranspose,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from utils import get_fisher

from utils.tf import (
    TimedLoggingCallback,
    dice_coefficient_loss,
)

from utils.tf.layers import (
    ReflectionPadding2D,
    create_context_module,
    create_convolution_block,
    create_up_sampling_module,
    create_localization_module,
    augmentation_layer,
)

from utils.tf.plots import (
    plot_metrics,
    plot_predictions,
    plot_histogram,
)

from dataloaders import TFDSDataLoader, DataLoader
from utils import SimConfig


def isensee_attn(
    inputs,
    depth=5,
    dropout_rate=0.3,
    n_segmentation_levels=3,
    n_labels=8,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="",
    metrics=[],
    interpolation="nearest",
    kernel_regularizer=None,
    flip=True,
    rotate=True,
    add_powers=0,
):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param inputs:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """


    input_layer = augmentation_layer(flip, rotate, add_powers)(inputs)

    current_layer = input_layer
    level_output_layers = []
    level_filters = []
    for level in range(depth):
        n_level_filters = 2 ** (4 + level // 2)
        level_filters.append(n_level_filters)

        if current_layer is input_layer:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(
                layer,
                n_level_filters,
                strides=(2, 2),
            )

        context_output_layer = create_context_module(
            in_conv,
            n_level_filters,
            dropout_rate=dropout_rate,
        )

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = []
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number], interpolation=interpolation
        )

        # Reg attention
        # with t2 this does not match sizes, need to fix
        attention = Attention()([level_output_layers[level_number], up_sampling])
        attention = Multiply()([up_sampling, attention])
        attention = LayerNormalization()(attention)

        concatenation_layer = Concatenate()(  # changed to add
            [
                level_output_layers[level_number],
                attention,
                # up_sampling,
            ]
        )
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number]
        )
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(
                0, Conv2D(1, (1, 1))(current_layer)
            )

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels - 1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            # output_layer = UpSampling2D(size=(2, 2), interpolation=interpolation)(
            #     output_layer
            # )
            output_layer = Conv2DTranspose(level_filters[level_number], kernel_size=(2, 2), strides=(2, 2))(output_layer)


    out_layer = Flatten()(output_layer)
    out_layer = Dropout(dropout_rate)(out_layer)
    out_layer = Dense(1)(out_layer) * 1000

    model = Model(inputs=inputs, outputs=out_layer, name=name)

    # Allows for us to pass in a complete optimizer or incomplete with learning rate
    if callable(optimizer):
        optimizer = optimizer(learning_rate=initial_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model


if __name__ == "__main__":
    # Get the config file from the command line
    # you can manually set it here if you want
    config_file = sys.argv[1]
    s = SimConfig(config_file)

    MAX_EPOCHS = 30

    # lower nside can not handle as deep of a network
    # higher nside will run out of memory
    DEPTH = 7
    N_SEG_LEVELS = 5

    # helps with the script running so we dont need to manually change the batch size
    if s.nside <= 128:
        BATCH_SIZE = 128
    elif s.nside <= 256:
        BATCH_SIZE = 64
    elif s.nside <= 512:
        BATCH_SIZE = 32
    elif s.nside <= 1024:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 1

    # just some info for the model name
    timestamp = int(time.time())
    lens_str = "lensed" if s.lensing else "unlensed"

    # Just gather some more info for the wandb logs
    extra_info = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID") or 0,
        "start_time": timestamp,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "comment": """new filters, no img norm, one label, full aug, 2x sigmoid, double batch size, changed depth and seg levels, adding res instead of concat""",
    }

    model_settings = {
        "depth": DEPTH,
        "n_segmentation_levels": N_SEG_LEVELS,
        "dropout_rate": 0.3,
        # "loss_function": tf.keras.losses.mse,
        "initial_learning_rate": 0.001,
        "name": f"{extra_info['slurm_job_id']}_{s.base_name}-{timestamp}",
        "n_labels": 16,
        "interpolation": "nearest",
        "kernel_regularizer": l2(1e-6),
        "flip": True,
        "rotate": True,
        "add_powers": 2,
    }

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error", "mse"]

    # enable a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=model_settings["initial_learning_rate"],
        decay_steps=100000,
        decay_rate=0.95,
        staircase=True,
    )

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=40,
        ),
        # model checkpoining to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
        TimedLoggingCallback(),
    ]

    # enable wandb, set to false if not using
    if False:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

        wandb.init(
            project="mlpng",
            notes=extra_info["comment"],
            tags=["isensee-attn", "dev"],
            config=s.settings | model_settings | extra_info,
            dir="data",
            sync_tensorboard=True,
        )

        # Add the wandb logger to the callbacks, so it is used
        callbacks.append(WandbMetricsLogger())

    # strategy lets us use multigpu,
    # if using a single gpu or no gpu it should do nothing
    # you can safely remove these 2 lines of code to disable this
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_img = Input((s.nside, s.nside, 1), name="input_image")

        opt = Adam(
            learning_rate=lr_schedule,
            # learning_rate=model_settings["initial_learning_rate"],
        )

        model = isensee_attn(
            input_img, optimizer=opt, metrics=metrics, **model_settings
        )

    # Lets load our data
    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "normalize": True,
        "batch_size": BATCH_SIZE,
        "cache": True,
    }
    tfds_filepath = s.data_file_complete.replace(".hdf5", ".tfds")
    if os.path.exists(tfds_filepath):
        # This might have a small speedup, but it also might not
        # This WILL let us run on multinode which the hdf5 loader does not
        data_loader = TFDSDataLoader(tfds_filepath, **data_loader_args)
    else:
        # load data as a python generator, directly from hdf5
        # This requires everything to be in the same python environment
        # aka, no multinode
        data_loader = DataLoader(s.data_file_complete, **data_loader_args)

    # Just print some good info for the log
    print("TensorFlow version:", tf.__version__)
    print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    print("Number of GPUs being used:", strategy.num_replicas_in_sync)

    pprint.PrettyPrinter(indent=2).pprint(model_settings | extra_info)
    print(data_loader)

    model.summary()

    # Split into train, test, and validation sets
    # We have enought data that we just use a 80/10/10 split
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    # Finally, lets fit our model
    # we use the train and val sets here, so the model will not see the test set
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    # Since we are doing sky cuts we expect the variance to be a bit higher than the fisher
    # Just get that here so we can plot it
    full_sky_degree = (np.pi / 180) ** 2
    f_sky = (s.settings["patch_side_deg"]) ** 2 / full_sky_degree
    fisher = get_fisher(s.data_file_complete)
    scaled_variance = np.sqrt(1 / (f_sky * fisher))

    # Lets plot the predictions from the unseen test set
    y_pred = model.predict(test_dataset, verbose=0)
    y_test = np.concatenate([y.numpy() for _, y in test_dataset])

    # Plot the loss curves and metrics
    plot_metrics(
        history,
        f"{s.plot_dir}/{model_settings['name']}-metrics.png",
        metrics=["loss"] + metrics,
    )
    plot_predictions(
        y_test,
        y_pred,
        f"{s.plot_dir}/{model_settings['name']}-preds.png",
        fisher,
        scaled_variance,
    )
    plot_histogram(
        y_test, y_pred, f"{s.plot_dir}/{model_settings['name']}-histogram.png"
    )
