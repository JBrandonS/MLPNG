import os
import tensorflow as tf
from keras import backend as K
from tensorflow import pad
from keras.layers import (
    Layer,
    UpSampling2D,
    SpatialDropout2D,
    Conv2D,
    BatchNormalization,
    GroupNormalization,
    Activation,
    Lambda,
)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import h5py


@tf.function(jit_compile=True)
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


@tf.function(jit_compile=True)
def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")


def create_localization_module(input_layer, n_filters):
    """
    This function creates a localization module, which is typically used in the
    decoder part of a U-Net architecture. It first applies reflection padding to
    the input, then applies two convolutional blocks. The second convolutional block
    uses a 1x1 kernel, which is often used to map each
    64-component feature vector to the desired number of classes.
    """
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    return create_convolution_block(convolution1, n_filters, kernel=(1, 1))


def create_up_sampling_module(
    input_layer, n_filters, size=(2, 2), interpolation="nearest"
):
    """
    This function creates an up-sampling module, which is used to increase the spatial
    dimensions of the input. It first applies up-sampling to the input using nearest
    neighbor interpolation, then applies reflection padding and a convolutional block.
    """
    up_sample = UpSampling2D(size=size, interpolation=interpolation)(input_layer)
    layer1 = ReflectionPadding2D()(up_sample)
    return create_convolution_block(layer1, n_filters)


def create_context_module(
    input_layer, n_level_filters, dropout_rate=0.1, data_format="channels_last"
):
    """
    This function creates a context module, which is typically used in the encoder part
    of a U-Net architecture. It first applies reflection padding to the input, then applies
    a convolutional block, a spatial dropout layer (which randomly sets entire feature maps
    to zero), another reflection padding layer, and another convolutional block.
    """
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(
        input_layer=layer1, n_filters=n_level_filters
    )
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = ReflectionPadding2D()(dropout)
    return create_convolution_block(input_layer=layer2, n_filters=n_level_filters)


def create_convolution_block(
    input_layer,
    n_filters,
    batch_normalization=False,
    kernel=(3, 3),
    activation="relu",  # "relu"
    kernel_initializer="he_uniform",
    padding="valid",
    strides=(1, 1),
    instance_normalization=True,
    kernel_regularizer=None,
):
    """
    This function creates a convolutional block, which is a fundamental building block of CNNs.
    It applies a 2D convolution to the input, followed by either batch normalization or instance
    normalization (if enabled). The number of filters, kernel size, activation function, padding,
    strides, and kernel initializer for the convolution can be specified as parameters.
    """
    layer = Conv2D(
        n_filters,
        kernel,
        padding=padding,
        strides=strides,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        # activation=activation,
    )(input_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        layer = GroupNormalization(groups=n_filters)(layer)
    layer = Activation(activation=activation)(layer)
    return layer


def rotation_layer(input_tensor):
    """
    rotates each image by a random number of 90 degree turns
    """

    def _work(image):
        k = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        return tf.image.rot90(image, k)

    rotated_tensor = Lambda(
        lambda x: _work(x), output_shape=input_tensor.shape, name="rotation_layer"
    )(input_tensor)
    return rotated_tensor


def plot_predictions(y_val, y_pred, save_file, fisher=None, scaled_variance=None):
    df = pd.DataFrame(
        {"True Labels": y_val.flatten(), "Predicted Labels": y_pred.flatten()}
    )

    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="True Labels", y="Predicted Labels")

    # Truth line
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
            label="Fisher",
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
            label="Scaled Variance (1/$\sqrt{f_{sky} f}$)",
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

    plt.title("Predicted vs True Labels")
    plt.savefig(save_file)


def plot_histogram(y_val, y_pred, save_file):
    """Plot and save a histogram of predictions with mean and std dev as title"""
    # Calculate mean and standard deviation
    y_val = y_val.flatten()
    y_pred = y_pred.flatten()

    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)

    diff = y_pred - y_val
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, figsize=(12, 12))

    # Plot the predictions on the first subplot
    sns.histplot(y_pred, ax=axs[0], legend=False)
    axs[0].set_title(f"Predictions - Mean: {mean_pred:.2f}, Standard Deviation: {std_pred:.2f}")

    # Plot the differences on the second subplot
    sns.histplot(diff, ax=axs[1], legend=False)
    axs[1].set_title(f"Differences - Mean: {mean_diff:.2f}, Standard Deviation: {std_diff:.2f}")

    # Save the plot
    plt.savefig(save_file)


def plot_metrics(history, save_file, metrics=["loss"]):
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, figsize=(15, 6 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history[f"val_{metric}"])
        axs[i].set_title(f"Model {metric}")
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Epoch")
        axs[i].legend(["Train", "Validation"], loc="upper right")

    plt.savefig(save_file)


def get_fisher(data_file):
    """Get the fisher matrix from the data file"""
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        kv = hdf.get("fisher", None)
        if kv is None:
            raise ValueError(f"Key fisher not found in {data_file}")
        return np.array(kv[()])  # type: ignore

def plot_activations(model, ds, name, layers=None, plot_dir="data/plots/activations"):
    import keract  # pip install keract for this to work

    first_batch = next(iter(ds.take(1)))
    images, _ = first_batch
    img = images[0][None, :, :, :]  # need to add back in the batch dim

    activations = keract.get_activations(model, img, auto_compile=True)

    if layers is not None:
        activations = activations.get(layers)

    keract.display_activations(
        activations,
        save=True,
        directory=os.path.join(plot_dir, name),
        data_format="channels_last",
    )