import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

import tensorflow as tf
import numpy as np

# Set up the MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
print(f"Python executable: {sys.executable}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")


# Define a simple model
def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


# Generate dummy data
num_samples = 10000
num_features = 784
num_classes = 10

x_train = np.random.random((num_samples, num_features))
y_train = np.random.randint(num_classes, size=(num_samples,))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# Create a dataset
batch_size = 8
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Distribute the dataset
dist_dataset = strategy.experimental_distribute_dataset(dataset)

# Train the model
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fit the model
    model.fit(
        dist_dataset, epochs=10, steps_per_epoch=num_samples // batch_size // 10 - 1
    )

print("Training complete.")
