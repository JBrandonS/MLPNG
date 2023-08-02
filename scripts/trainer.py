# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 1
os.environ["TF_XLA_FLAGS"]="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"

import tensorflow as tf

keras = tf.keras  # fixes issues with vsCode
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13})

from keras import Input, Model
from keras.layers import (
    Add,
    Flatten,
    Dense,
    RandomRotation,
    RandomFlip,
)
from keras.optimizers import Adam

from datetime import datetime

from keras.callbacks import (
    TensorBoard,
    EarlyStopping,
)

from dataloader import DataLoader

from isensee import *
import wandb

# %%
# tf.debugging.experimental.enable_dump_debug_info('data/tensorboard', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# tf.debugging.experimental.disable_dump_debug_info()

# %matplotlib inline
print(f"tf version: {tf.__version__}")

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

# %%
nside = 128
no_noise = False
pols = 'T'
nsims = 1000
lensed = False

npatches = 10
fnl_range = [-1000, 1000]

batch_size = 128 # 2**n for GPU
max_epochs = 100
save_models=True
debug = True
use_tensorboard = True

# holds base settings for wandb
wandb_config = {
    'nside': nside,
    'nsims': nsims,
    'pols': pols,
    'lensed': lensed,
    'no_noise': no_noise,
    'fnl_range': fnl_range,
    'batch_size': batch_size,
    'max_epochs': max_epochs
}

# %%
def mk_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating dir: {dir}")
        try:
            os.makedirs(dir)
        except FileExistsError:
            # race condition, can happen in job arrays
            pass
    else:
        print(f'Using existing folder: {dir}')

npol = len(pols)

# some base settings for files
lensed_str = 'lensed' if lensed else 'unlensed'
nn_str = '_nn' if no_noise else ''

basename = f'{nside}{nn_str}_{pols}_{nsims}'
data_filename = f'{basename}x{npatches}_fnl{fnl_range[0]}-{fnl_range[1]}'

run_time = datetime.now().strftime("%Y%m%d-%H%M%S")

data_dir = f"data/{lensed_str}"
model_dir = f"data/models"
plot_dir = f"data/plots"
tb_dir = f"data/tensorboard"

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
print('data sizes', train_size, val_size, test_size)

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
    ret = ret.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, name=name)
    ret = ret.prefetch(tf.data.AUTOTUNE)
    return ret

train_dataset = get_data(0, train_size, 'train')
val_dataset = get_data(train_size, val_size, 'val')
test_dataset = get_data(train_size + val_size, test_size, 'test')

# %% [markdown]
# ## Train Model
# ### Train the Isensee Model

# %%
with strategy.scope():
    model_settings = {
        'depth':4,
        'n_segmentation_levels':4,
        'dropout_rate':0.03,
        'loss_function':tf.keras.losses.mse,
        'initial_learning_rate':0.01,
        'name':f"isensee-{basename}",
    }
    wandb.init(project="mlpng", 
               notes="isensee", 
               tags=["isensee", "dev"], 
               reinit=True, 
               tensorboard=True, 
            #    sync_tensorboard=True, 
               config= wandb_config | model_settings)

    input_img = Input((nside, nside, 1), name="img")
    isensee_model = isensee2017_model(input_img, **model_settings)
    
    if debug:
        isensee_model.summary()

    callbacks = [
        # EarlyStopping(monitor="val_root_mean_square_error", patience=10, verbose=1, restore_best_weights=True),
        # ModelCheckpoint(
        #     filepath=f"{model_dir}/checkpoint_{run_time}.keras",
        #     monitor="val_loss",
        #     save_best_only=True,
        # ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_square_error',factor=0.1,patience=5),
        wandb.keras.WandbMetricsLogger(),
        wandb.keras.WandbModelCheckpoint(filepath=f"{model_dir}/wandb"),
    ]
    if use_tensorboard:
        callbacks.append(TensorBoard(log_dir=tb_dir))

    isensee_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        callbacks=callbacks,
    )

    isensee_model.predict(test_dataset, batch_size=batch_size, verbose="auto", callbacks=callbacks, use_multiprocessing=True)

    wandb.finish(0)

# %% [markdown]
# ### Train the BS model

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
        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.KLDivergence()],
    )
    return model

# %%
with strategy.scope():
    input_img = Input((nside, nside, 1), name="img")
    model_settings = {
        'depth':4,
        'dropout_rate':0.3,
        'loss_function':tf.keras.losses.mse,
        'initial_learning_rate':0.01,
        'preprocess':False,
        'name':f"bs-{basename}",
    }
    wandb.init(project="mlpng", 
               notes="bs", 
               tags=["bs", "dev"], 
               reinit=True, 
               config=wandb_config | model_settings)
    bs_model = make_bs_model(input_img, **model_settings)
    
    if debug:
        bs_model.summary()

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

    bs_model.predict(test_dataset, batch_size=batch_size, verbose="auto", callbacks=callbacks)

# %%
if not debug:
    print('Done Training!')
    # exit(0)

# %% [markdown]
# # Validation
#
# Simple validation checks on the three datasets. The model has seen train and val before, so those results are not as important as the test results which have not been seen by the model yet.
# 
# TODO: Once we have added FI, plot the FI for each dataset.

# %%
def plt_pred(dataset, name, save=False):
    ipreds = isensee_model.predict(dataset, batch_size=batch_size, verbose="auto", callbacks=callbacks)[:, 0]
    bpreds = bs_model.predict(dataset, batch_size=batch_size, verbose="auto", callbacks=callbacks)[:, 0]

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
# Model has not seen the test data, so this is the best view of performance.

# %%
plt_pred(test_dataset, 'test', True)


