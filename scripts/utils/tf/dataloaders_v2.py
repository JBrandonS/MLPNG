import logging

import numpy as np
from numpy.random import default_rng

import tensorflow as tf
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras import Input

import tensorflow_io as tfio

logger = logging.getLogger(__name__)


class DataLoaderBase_v2:

    def __init__(
        self,
        file_path,
        x_dataset="/patch",
        y_dataset="/fnl",
        name=None,
        batch_size=1,
        shuffle=False,
        shuffle_buffer=1000,
        normalize=False,
        cache=False,
        seed=None,
        preprocess=True,
        dtype=np.float32,
        num_replicas="auto",
    ):
        logger.debug(f"Creating DataLoaderBase_v2 with file_path: {file_path}")
        # setup some attributes
        self.file_path = file_path
        self.name = self.__class__.__name__ if name is None else name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.cache = cache
        self.normalize = normalize
        self.seed = seed if seed is not None else default_rng().integers(0, 2**32 - 1)
        self.dtype = dtype
        self.preprocess = preprocess

        if num_replicas == "auto":
            # tested on superpod but would not be surprised if this doesn't work on other systems
            self.num_replicas = len(tf.config.list_physical_devices("GPU")) or 1
            logger.debug(f"auto detected num_replicas: {self.num_replicas}")
        else:
            # make sure num_replicas is >= 1
            self.num_replicas = int(num_replicas) if int(num_replicas) > 0 else 1

        x_ds = tfio.IODataset.from_hdf5(file_path, x_dataset)
        y_ds = tfio.IODataset.from_hdf5(file_path, y_dataset)
        self._ds = Dataset.zip((x_ds, y_ds), name=name)

    def _preprocess(self, data, y):
        """Used to convert the data to the correct shape"""

        # by default we deal with patch data of shape (pol, patches, nside, nside)
        # but we want to convert this to (patches, nside, nside, pol) so that pol can act as features

        data = tf.transpose(data, [1, 2, 3, 0])
        y = tf.transpose(y, [1, 0])  # move pol to the last dimension

        # we do need to repeat the FNL value, which will be the same for all patches
        y = tf.repeat(y, data.shape[0], axis=0)

        # data = tf.stack(data)
        return (data, y)

    def _get_dataset(self, start, step):
        """
        Creates a subset of the dataset starting from the 'start' index and taking 'step' number of elements.

        The subset is processed according to the class's settings: it is normalized if 'normalize' is True,
        cached if 'cache' is True, and shuffled if 'shuffle' is True. The subset is then batched with
        'batch_size' number of elements per batch.

        Parameters:
        start (int): The index to start the subset from.
        step (int): The number of elements to include in the subset.

        Returns:
        tf.data.Dataset: The processed subset of the dataset, ready for training or evaluation.
        """
        data = self._ds.skip(start).take(step)

        # if self.preprocess:
        #     data = data.map(self._preprocess)  # , num_parallel_calls=AUTOTUNE)

        #     # we unbatch the dataset so that each patch is seen as a element
        #     data = data.unbatch()
        #     logger.debug(f"data {data}")

        # fixes an issue with tf not knowing the size of the dataset
        # data = data.apply(tf.data.experimental.assert_cardinality(10 * step))

        # if self.normalize:
        #     data = data.map(self._normalize, num_parallel_calls=AUTOTUNE)

        # if self.cache:
        #     data = data.cache()

        # if self.shuffle:
        #     buffer_size = step if self.shuffle_buffer is None else self.shuffle_buffer
        #     data = data.shuffle(buffer_size, self.seed, reshuffle_each_iteration=True)

        data = data.batch(
            # with keras, self.batch_size is local batch size, but we want this to be global batch size
            # to convert we just multiply the local by the number of replicas
            # see: https://keras.io/guides/distributed_training_with_tensorflow/
            # self.batch_size * self.num_replicas,
            # drop_remainder=True,  # we dont want any partial batches, this is to support XLA opts
            # deterministic=False,  # we dont care about order
            # num_parallel_calls=AUTOTUNE,
            1
        )

        return data  # .prefetch(AUTOTUNE)

    def get_split(self, train_frac, test_frac, val_frac):
        """
        Splits the dataset into training, testing, and validation sets.

        This method calculates the sizes of the training and testing sets based on the provided fractions,
        and then creates the actual datasets.

        Parameters:
        train_frac (float): The fraction of the dataset to use for training.
        test_frac (float): The fraction of the dataset to use for testing.
        val_frac (float): The fraction of the dataset to use for validation.

        Returns:
        tuple: The training and testing datasets, and the validation dataset.
        """
        train_size = int(self.length * train_frac)
        test_size = int(self.length * test_frac)
        val_size = int(self.length * val_frac)

        logger.debug(
            f"Dataset sizes: Training {train_size}, Testing {test_size}, Validation {val_size}"
        )

        train_ds = self._get_dataset(0, train_size)
        test_ds = self._get_dataset(train_size, test_size)
        val_ds = self._get_dataset(train_size + test_size, val_size)
        return train_ds, test_ds, val_ds

    def input(self):
        """returns a keras input for the dataset"""
        return Input(self.shape)

    def _normalize(self, data, label):
        """
        Normalize the input data by subtracting the mean and dividing by the standard deviation.
        This method treats the 0's as a mask thus not counting the empty data, which can be useful for sparse data.

        Args:
            data (tf.Tensor): The input data tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            tf.Tensor: The normalized data tensor.
            tf.Tensor: The label tensor.
        """
        mask = data != 0
        masked_data = tf.boolean_mask(data, mask)

        mean = tf.math.reduce_mean(masked_data)
        std = tf.math.reduce_std(masked_data)

        data = tf.where(mask, (data - mean) / std, data)
        return data, label
