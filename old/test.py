# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"


import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Embedding, Input, Concatenate
from tensorflow.keras.layers import Subtract, Attention
from tensorflow.keras.utils import plot_model
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras import backend as K

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from tensorflow.keras.layers import Multiply, Add, Lambda, GroupNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from tensorflow.keras.layers import (
    Layer,
    LeakyReLU,
    PReLU,
    Add,
    UpSampling2D,
    Activation,
    SpatialDropout2D,
    Conv2D,
    SeparableConv2D,
    BatchNormalization,
    Concatenate,
    Flatten,
    Dense,
)

from tensorflow import pad

from scripts.config import SimConfig
from scripts.dataloader import DataLoader

import seaborn as sns
import pandas as pd

from sklearn import metrics as mt
from matplotlib import pyplot as plt
# %matplotlib inline

import h5py

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

# %%
s = SimConfig("settings/ul_nn_256.json")

# load data as a generator so we do not need to have it all in memory
d = tf.data.Dataset.from_generator(
    DataLoader(s.data_file_complete, shuffle=True, normalize=False),
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

# %%
def get_data(start, step, name=None):
    ret = d.skip(start).take(step)

    # # This just fixes the logging output not knowing the dataset size
    ret = ret.apply(tf.data.experimental.assert_cardinality(step))

    ret = ret.cache()
    # ret = ret.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, name=name)
    ret = ret.prefetch(tf.data.AUTOTUNE)
    return ret

# %%

# Setup the data split as 80/10/10
n = s.total_sims

train_size = int(n * 0.8)
val_size = int(n * 0.1)
test_size = int(n * 0.1)

# loads in the datasets
train_dataset = get_data(0, train_size, "train")
val_dataset = get_data(train_size, val_size, "val")
test_dataset = get_data(train_size + val_size, test_size, "test")

print('Training size:', train_size, 'Validation size:', val_size, 'Test size:', test_size)


# %%
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

# %%
IMG_SHAPE = (s.nside, s.nside, 1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

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
    activation="sigmoid", # maybe try PReLU
    padding="valid",
    strides=(1, 1),
    instance_normalization=True,
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
        layer = GroupNormalization(groups=n_filters)(layer)
        # layer = InstanceNormalization()(layer)
    layer = Activation(activation)(layer)
    return layer

def attn_model(
    inputs,
    depth=8,
    dropout_rate=0.3,
    n_segmentation_levels=7,
    n_labels=8,
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
    n_level_filters = 32
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

        attention1 = Attention(use_scale=True, dropout=dropout_rate)([level_output_layers[level_number], up_sampling])

        att_layer = Add()([attention1, up_sampling])
        att_conv = create_convolution_block(att_layer, level_filters[level_number], padding='same')

        concatenation_layer = Concatenate()(
            [level_output_layers[level_number], att_conv]
        )
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number]
        )
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(16, (1, 1), activation="sigmoid")(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels - 1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2))(output_layer)

    out_layer = Flatten()(output_layer)
    out_layer = Dropout(dropout_rate)(out_layer)
    out_layer = Activation("sigmoid")(out_layer)
    out_layer = Dense(1024)(out_layer)
    out_layer = Dense(1)(out_layer)

    model = Model(inputs=inputs, outputs=out_layer, name=name)
    return model

# %%
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = attn_model(Input(shape=IMG_SHAPE), name='attn_model')

    model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error'])
    
print(model.summary())

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='test-model.png', show_shapes=True, show_layer_names=True, expand_nested=True, show_layer_activations=True, show_trainable=True)

# early_stopping = EarlyStopping(monitor='val_loss', patience=30)

lr_decay = 0.5
lrd_patience = 10 #10, 3, 20 (originally 3)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=lr_decay, patience=lrd_patience)

tb = TensorBoard(log_dir='./logs/tensorboard')

NUM_EPOCHS = 300
history = model.fit(X_train, 
                    y_train, 
                    epochs=NUM_EPOCHS,
                    verbose = 2, 
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, tb])

# %%

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()
plt.savefig(f'{s.plot_dir}/2dmse-e{NUM_EPOCHS}-{s.base_name}-loss.png')


# %%

# Predict on the validation dataset
y_pred = model.predict(X_val, verbose=1)

# Create a dataframe with true and predicted labels
df = pd.DataFrame({'True Labels': y_val.flatten(), 'Predicted Labels': y_pred.flatten()})
mean = df['True Labels'].mean()

fisher = load_data(s.data_file_complete, ["fisher"], verbose=s.verbose)["fisher"]
std_dev = np.sqrt(1/fisher)

from sklearn.metrics import r2_score
r2 = r2_score(df['True Labels'], df['Predicted Labels'])

# Create a scatter plot with seaborn
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='True Labels', y='Predicted Labels')

# plt.plot([min(y_val), max(y_val)], [mean, mean], color='green', linestyle='--')
# plt.plot([min(y_val), max(y_val)], [mean + std_dev, mean + std_dev], color='blue', linestyle='--')
# plt.plot([min(y_val), max(y_val)], [mean - std_dev, mean - std_dev], color='blue', linestyle='--')

# Line for perfect fit
plt.text(min(y_val), max(y_val), f'R^2 = {r2:.2f}', verticalalignment='top')

plt.title(s.base_name)

plt.tight_layout()
plt.show()
plt.savefig(f'{s.plot_dir}/2dmse-e{NUM_EPOCHS}-{s.base_name}.png')


