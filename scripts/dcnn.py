import math
import os
import pprint
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.environ['CUDA_HOME']}"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add,
    Attention,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Multiply,
    RandomFlip,
    SpatialDropout2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

from dataloaders import DataLoader, TFDSDataLoader
from utils import SimConfig
from utils.tf import (
    ReflectionPadding2D,
    create_context_module,
    create_convolution_block,
    dice_coefficient_loss,
    rotation_layer,
    TimedLoggingCallback,
    get_fisher,
    plot_histogram,
    plot_metrics,
    plot_predictions,
)


def deep_cnn_block(
    input_layer,
    n_filters,
    depth=2,
    kernel=(3, 3),
    activation="sigmoid",
    kernel_initializer="glorot_uniform",
    padding="valid",
    strides=(1, 1),
    kernel_regularizer=None,
    avg_pooling=False,
    max_pooling=True,
    residual=False,
    attention=True,
):
    layer = input_layer

    # create a CNN block
    for _ in range(2):
        for level in range(2):
            if level < depth - 1:
                layer = Conv2D(
                    n_filters,
                    kernel,
                    padding=padding,
                    strides=strides,
                    kernel_initializer=kernel_initializer,
                    # kernel_regularizer=kernel_regularizer,
                    activation=activation,
                )(layer)
            else:
                # no activation on last layer
                layer = Conv2D(
                    n_filters,
                    kernel,
                    padding=padding,
                    strides=strides,
                    # kernel_regularizer=kernel_regularizer,
                )(layer)

            layer = BatchNormalization()(layer)

            # restore the shape of the input layer
            pad = (kernel[0] - 1) // 2
            layer = ReflectionPadding2D((pad, pad))(layer)

    if attention:

        gate = Conv2D(
            n_filters,
            (1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
        )(input_layer)
        att_layer = Attention(use_scale=True)([gate, layer])
        layer = Multiply()([att_layer, layer])

    if residual:
        # this is a 1x1 convolution to match the number of filters
        res_layer = Conv2D(
            n_filters,
            (1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
        )(input_layer)
        layer = Add()([layer, res_layer])
        # layer = Concatenate()([layer+res_layer, layer, res_layer])

    # Pool at the end to reduce size
    if avg_pooling:
        layer = AveragePooling2D(pool_size=(2, 2))(layer)
    elif max_pooling:
        layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # finally normalize the layer
    layer = BatchNormalization()(layer)

    return layer


def dcnn_model(
    inputs,
    optimizer=Adam,
    metrics=[],
    depth=3,
    dropout_rate=0.3,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="",
    kernel_regularizer=None,
    flip=False,
    rotate=False,
    add_t2=False,
    ff_activation="relu",
    ff_kinit="he_uniform",
    **kwargs,
):
    input_layer = inputs

    # this finds the number of factor of 2 reductions in the spatial dimensions to make final depth 16x16
    # just to ensure we don't go too deep / small
    max_depth = math.log(inputs.shape[1] / 32, 2) + 1
    depth = min(depth, int(max_depth))

    if flip:
        input_layer = RandomFlip()(input_layer)

    if rotate:
        # random rotation
        input_layer = rotation_layer(input_layer)

    if add_t2:
        # Squares every pixel and add them, gets T2 map
        squared = tf.square(input_layer)
        cube = tf.pow(input_layer, 3)
        input_layer = Concatenate()([input_layer, squared, cube])

    layer = input_layer
    for level in range(depth):
        n_level_filters = 2 ** (4 + 2 * level)

        layer = deep_cnn_block(
            layer,
            n_level_filters,
            kernel_regularizer=kernel_regularizer,
        )

        layer = SpatialDropout2D(dropout_rate)(layer)

    # FF network
    out_layer = Flatten()(layer)
    out_layer = Dropout(dropout_rate)(out_layer)

    min_neurons = 32
    n_neurons = out_layer.shape[-1]
    while n_neurons > 1:
        n_neurons = max(1, n_neurons // min_neurons)

        if n_neurons < min_neurons:
            # create last layer with 1 neuron, and no activation then stop
            out_layer = Dense(1, activation="sigmoid")(out_layer)
            break
        else:
            out_layer = Dense(
                n_neurons,
                activation=ff_activation,
                kernel_initializer=ff_kinit,
                # kernel_regularizer=kernel_regularizer,
            )(out_layer)

    # allows the model to operate on a -1,1 scale
    out_layer = out_layer * 1000

    model = Model(inputs=inputs, outputs=out_layer, name=name)

    # Allows for us to pass in a complete optimizer or incomplete with learning rate
    if callable(optimizer):
        optimizer = optimizer(learning_rate=initial_learning_rate)

    # finally we compile the model
    # needs to be done in strategy scope
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

    MAX_EPOCHS = 300

    DEPTH = 2

    # helps with the script running so we dont need to manually change the batch size
    if s.nside <= 128:
        BATCH_SIZE = 128
    elif s.nside <= 256:
        BATCH_SIZE = 64
    elif s.nside <= 512:
        BATCH_SIZE = 32
    elif s.nside <= 1024:
        BATCH_SIZE = 1
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
        "comment": """deep cnn test""",
    }

    model_settings = {
        "depth": DEPTH,
        "dropout_rate": 0.3,
        "loss_function": tf.keras.losses.mse,
        "initial_learning_rate": 5e-5,
        "name": f"{extra_info['slurm_job_id']}_DeepCCN_{s.base_name}-{timestamp}",
        "kernel_regularizer": l2(1e-6),
    }

    data_loader_args = {
        "shuffle": True,
        "seed": None,
        "normalize": True,
        "batch_size": BATCH_SIZE,
        "cache": True,
    }

    # additional metrics we are intrested in
    metrics = ["mean_absolute_error"]

    # enable a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=model_settings["initial_learning_rate"],
        decay_steps=10000,
        decay_rate=0.95,
        staircase=True,
    )

    # callbacks to use during training
    callbacks = [
        # We use earlystoping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=0,
        ),
        # model checkpoining to save the best model
        ModelCheckpoint(
            f"{s.model_dir}/{model_settings['name']}" + "-{epoch:03d}.tf",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        ),
        TimedLoggingCallback(),  # custom logger to work a little better with text logs
    ]

    # enable wandb, set to false if not using
    if True:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

        wandb.init(
            project="mlpng",
            notes=extra_info["comment"],
            tags=["DeepCNN", "dev"],
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
            # learning_rate=burnin_schedule(model_settings["initial_learning_rate"]),
            # learning_rate=lr_schedule,
            learning_rate=model_settings["initial_learning_rate"],
        )

        model = dcnn_model(input_img, opt, metrics, **model_settings)

    # Lets load our data
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
    # using small dataset rn
    train_dataset, test_dataset, val_dataset = data_loader.get_split(0.8, 0.1, 0.1)

    # Finally, lets fit our model
    # we use the train and val sets here, so the model will not see the test set
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=0,  # since we are using the custom logger
    )

    # Lets plot the predictions from the unseen test set
    y_pred = model.predict(test_dataset, verbose=0)
    y_test = np.concatenate([y.numpy() for _, y in test_dataset])

    # Since we are doing sky cuts we expect the variance to be a bit higher than the fisher
    # Just get that here so we can plot it
    full_sky_degree = (np.pi / 180) ** 2
    f_sky = (s.settings["patch_side_deg"]) ** 2 / full_sky_degree
    fisher = get_fisher(s.data_file_complete)
    scaled_variance = np.sqrt(1 / (f_sky * fisher))

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
