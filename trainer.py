# %%
import os
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 1
os.environ["TF_XLA_FLAGS"]="--tf_xla_auto_jit=2"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"

import tensorflow as tf

keras = tf.keras  # fixes issues with vsCode
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13})

from functools import partial

from numpy import random

from keras import Input, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Layer,
    LeakyReLU,
    Add,
    UpSampling2D,
    Activation,
    SpatialDropout2D,
    Conv2D,
    BatchNormalization,
    Concatenate,
    Flatten,
    Dense,
    RandomRotation,
    RandomFlip,
    RandomZoom,
)
from keras.optimizers import Adam

from IPython.display import clear_output
from datetime import datetime

from tensorflow import pad
from tensorflow_addons.layers import InstanceNormalization

from keras.callbacks import (
    LambdaCallback,
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)

import h5py
import healpy as hp

from dataloader import DataLoader

# %matplotlib inline
print(f"tf version: {tf.__version__}")

# %%
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices("GPU")

print(gpus)

# %% [markdown]
# We used MirroredStrategy to allow for multiGPU setups, can use onedevice if you need to debug things and don't want to change server settings.
# 
# You need to use `with strategy.scope():` anytime you call a tensorflow function with this code or you will get errors.

# %%
strategy = tf.distribute.MirroredStrategy(gpus)
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") # for debugging

# %% [markdown]
# ### Helper Functions

# %%
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "padding": self.padding,
            "input_spec": self.input_spec,
        })
        return config


def create_localization_module(input_layer, n_filters):
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    return create_convolution_block(convolution1, n_filters, kernel=(1, 1))


def create_up_sampling_module(input_layer, n_filters, size=(2, 2)):
    up_sample = UpSampling2D(size=size)(input_layer)
    layer1 = ReflectionPadding2D()(up_sample)
    return create_convolution_block(layer1, n_filters)


def create_context_module(
    input_layer, n_level_filters, dropout_rate=0.1, data_format="channels_last"
):
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(
        input_layer=layer1, n_filters=n_level_filters
    )
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = ReflectionPadding2D()(dropout)
    return create_convolution_block(
        input_layer=layer2, n_filters=n_level_filters
    )


def create_convolution_block(
    input_layer,
    n_filters,
    batch_normalization=False,
    kernel=(3, 3),
    activation=LeakyReLU,
    padding="valid",
    strides=(1, 1),
    instance_normalization=False,
):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        layer = InstanceNormalization()(layer)
    return Activation("relu")(layer) if activation is None else activation()(layer)


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

with strategy.scope():
    create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

# %%
nside = 1024
no_noise = True
pols = 'T'
nsims = 1000

lensed = True
npatches = 10
fnl_range = [-1000, 1000]

use_tensorboard = True
batch_size = 8 # 2**n for GPU
max_epochs = 100

save_models=True
debug = True

# %%
def mk_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating dir: {dir}")
        os.makedirs(dir)
    else:
        print(f'Using existing folder: {dir}')

npol = len(pols)

# some base settings for files
model_name = "ksw"
lensed_str = 'lensed' if lensed else 'unlensed'
nn_str = '_nn' if no_noise else ''

data_dir = f"data/{model_name}/{lensed_str}"
model_dir = f"{data_dir}/models"
plot_dir = f"{data_dir}/plots"
tb_dir = f"{data_dir}/tensorboard"

basename = f'{nside}{nn_str}_{pols}_{nsims}'
data_filename = f'{basename}x{npatches}_fnl{fnl_range[0]}-{fnl_range[1]}'

run_time = datetime.now().strftime("%Y%m%d-%H%M%S")

print('Using data file', data_filename)

# %%
mk_dir(data_dir)
mk_dir(model_dir)
mk_dir(plot_dir)
if use_tensorboard:
    mk_dir(tb_dir)

# %%
n = npatches * nsims
train_size = int(n * 0.8)
val_size = int(n * 0.1)
test_size = int(n * 0.1)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

d = tf.data.Dataset.from_generator(
    DataLoader(f'{data_dir}/{data_filename}.hdf5'),
    output_signature=(tf.TensorSpec(shape=(nside, nside), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)),
)
d = d.with_options(options)

def get_data(start, step, name=None):
    ret = d.skip(start).take(step)

    # This just fixes the logging output not knowing the dataset size
    ret = ret.apply(tf.data.experimental.assert_cardinality(step))

    ret = ret.cache()
    ret = ret.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, name=name)
    ret = ret.prefetch(tf.data.AUTOTUNE)
    return ret

train_dataset = get_data(0, train_size, 'train')
val_dataset = get_data(train_size, val_size, 'val')
test_dataset = get_data(train_size + val_size, test_size, 'test')

print('data sizes', train_size, val_size, test_size)

# %% [markdown]
# ## Train Model

# %% [markdown]
# Set up callbacks for both models.

# %%
with strategy.scope():
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5, decay_steps=1000, decay_rate=0.9
    )

    callbacks = [
        EarlyStopping(monitor="val_root_mean_squared_error", patience=8),
        ModelCheckpoint(
            filepath=f"{model_dir}/checkpoint_{run_time}.keras",
            monitor="val_root_mean_squared_error",
            save_best_only=True,
        ),
        # LearningRateScheduler(lr_schedule),
    ]

    if use_tensorboard:
        tblog_dir = f"{tb_dir}/{basename}_{run_time}"
        callbacks.append(TensorBoard(log_dir=tblog_dir, write_images=True))

# %% [markdown]
# Train the isensee model

# %%
def isensee2017_model(
    inputs,
    n_base_filters=16,
    depth=5,
    dropout_rate=0.3,
    n_segmentation_levels=3,
    n_labels=1,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    activation_name="relu",
    name=''
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

    current_layer = inputs
    level_output_layers = []
    level_filters = []
    # n_level_filters = (2**level_number) * n_base_filters
    n_level_filters = 8
    for _ in range(depth):
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters, strides=(2, 2))

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate
        )

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = []
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number]
        )
        concatenation_layer = Concatenate()(
            [level_output_layers[level_number], up_sampling]
        )
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number]
        )
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels - 1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2))(output_layer)

    flat_layer = Flatten()(output_layer)
    out_layer = Dense(1, activation=None)(flat_layer)

    model = Model(inputs=inputs, outputs=out_layer, name=name)
    model.compile(
        optimizer=optimizer(learning_rate=initial_learning_rate),
        loss=loss_function,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )
    return model

# %%
with strategy.scope():
    input_img = Input((nside, nside, 1), name="img")
    isensee_model = isensee2017_model(
        input_img,
        depth=5,
        n_segmentation_levels=3,
        dropout_rate=0.3,
        loss_function=tf.keras.losses.mse,
        initial_learning_rate=1e-3,
        name=f"isensee-{basename}",
    )
    
    if debug:
        isensee_model.summary()

# %%
with strategy.scope(): 
    isensee_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        callbacks=callbacks,
    )

    if save_models:
        modelfile = f"{model_dir}/isensee_{run_time}"
        isensee_model.save(modelfile)

# %% [markdown]
# Train the BS model

# %%
def make_bs_model(
    inputs,
    depth=5,
    dropout_rate=0.3,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="custom_model",
    preprocess=False,
):
    if preprocess:
        inputs = RandomFlip("horizontal")(inputs)
        # inputs = RandomZoom(0.2)(inputs)
        inputs = RandomRotation(np.pi)(inputs)

    current_layer = inputs
    level_output_layers = []
    level_filters = []
    n_level_filters = 16 
    for _ in range(depth):
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters, strides=(2, 2))

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate
        )
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    flat_layer = Flatten()(current_layer)
    out_layer = Dense(1, activation=None)(flat_layer)

    model = Model(inputs=inputs, outputs=out_layer, name=name)
    model.compile(
        optimizer=optimizer(learning_rate=initial_learning_rate),
        loss=loss_function,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )
    return model

# %%
with strategy.scope():
    input_img = Input((nside, nside, 1), name="img")
    bs_model = make_bs_model(
        input_img,
        depth=6,
        dropout_rate=0.3,
        loss_function=tf.keras.losses.mse,
        initial_learning_rate=1e-3,
        preprocess=True,
        name=f"{model_name}-{basename}",
    )
    
    if debug:
        bs_model.summary()

# %%
with strategy.scope():
    bs_model.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=callbacks,
    )

    if save_models:
        modelfile = f"{model_dir}/bs_{run_time}"
        bs_model.save(modelfile)

# %%
if not debug:
    print('Done Training!')
    exit(0)

# %% [markdown]
# # Validation

# %% [markdown]
# Simple validation checks on the three datasets. The model has seen train and val before, so those results are not as important as the test results which have not been seen by the model yet.
# 
# TODO: Once we have added FI, plot the FI for each dataset.

# %%
def plt_pred(dataset, name, save=False):
    ipreds = isensee_model.predict(dataset, batch_size=batch_size, verbose="auto")[:, 0]
    bpreds = bs_model.predict(dataset, batch_size=batch_size, verbose="auto")[:, 0]

    truth = dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()
    truth = np.array(list(truth))


    irmse = np.sqrt(((ipreds - truth) ** 2).mean())
    brmse = np.sqrt(((bpreds - truth) ** 2).mean())

    plt.plot(truth, ipreds, ".", label=f'isensee: {irmse:.2f}')
    plt.plot(truth, bpreds, ".", label=f'bs: {brmse:.2f}')
    plt.plot(truth, truth, ".", label="truth")
    plt.title(f"{name} predictions")
    plt.xlabel(f"truth")
    plt.ylabel("prediction")

    plt.legend()
    plt.grid()
    plt.show()

    if save:
        plt.savefig(f"{plot_dir}/{basename}_{name}.png")

# %%
plt_pred(train_dataset, 'training', True)

# %%
plt_pred(val_dataset, 'validation', True)

# %% [markdown]
# Model has not seen the test data, so this is the best view of preformance.

# %%
plt_pred(test_dataset, 'test', True)


