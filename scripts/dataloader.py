import numpy as np
import h5py
import copy
from itertools import product

import logging

from tensorflow.keras.utils import Sequence
import tensorflow as tf


class DataLoader(Sequence):
    def __init__(self, file_name, shuffle=True, seed=None, normalize=False, batch_size=1, cache=True):
        """
        Initializes the DataLoader object.

        Args:
            file (str): The path to the file to load data from.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            seed (int, optional): The seed for the random number generator used for shuffling.
                If None, a random seed is generated. Defaults to None.
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
        """
        self.logger = logging.getLogger(__name__)
        self.file_name = file_name
        self.shuffle = shuffle
        self.normalize = normalize
        self.batch_size = batch_size
        self.cache = cache

        if seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            self.seed = seed

        self.file = h5py.File(self.file_name, mode="r", swmr=True, locking=False)

        (
            self._nsims,
            self._ndup,
            self._npol,
            self._npatches,
            self._nside,
            _,
        ) = self.file["patches"].shape

        print(self.file["patches"].shape)

        self.length = self._nsims * self._ndup * self._npol * self._npatches
        self.idxs = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def __str__(self):
        return (
            "DataLoader(file: %s, Seed: %s, Shuffle: %s, Normalize: %s, Length: %s, Batch Size: %s, Cache: %s)"
            % (self.file_name, self.seed, self.shuffle, self.normalize, self.length, self.batch_size, self.cache)
        )

    def __getitem__(self, index):
        # Convert the flat index to a multidimensional index
        i, j, k, l = np.unravel_index(
            index, (self._nsims, self._ndup, self._npol, self._npatches)
        )

        # we also need to add the channel dimension as TF expects it
        patch = self.file["patches"][i, j, k, l][:, :, None]
        fnl = self.file["fnls"][i, j]

        if self.normalize:
            min_val = np.min(patch)
            max_val = np.max(patch)
            patch = (patch - min_val) / (max_val - min_val)

        return np.array(patch), np.array(fnl)

    def __len__(self):
        return self.length

    # def __iter__(self):
    #     for index in self.idxs:
    #         i, j, k, l = np.unravel_index(
    #             index, (self._nsims, self._ndup, self._npol, self._npatches)
    #         )

    #         patch = self.file["patches"][i, j, k, l][:, :, None]
    #         if self.normalize:
    #             min_val = np.min(patch)
    #             max_val = np.max(patch)
    #             patch = (patch - min_val) / (max_val - min_val)

    #         fnl = self.file["fnls"][i, j]
    #         yield tf.convert_to_tensor(patch), tf.convert_to_tensor(fnl)

    def _setup_tfds(self, ds, start, step):
        ret = ds.skip(start).take(step)
        ret = ret.apply(tf.data.experimental.assert_cardinality(step))
        if self.cache:
            ret = ret.cache()
        if self.batch_size is not None and self.batch_size > 1:
            ret = ret.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        return ret.prefetch(tf.data.AUTOTUNE)

    def get_split_tfdataset(
        self, train_frac=0.8, test_frac=0.1, val_frac=0.1
    ):
        n = self.length
        train_size = int(n * train_frac)
        val_size = int(n * val_frac)
        test_size = int(n * test_frac)

        tfds = tf.data.Dataset.from_generator(
            lambda: self,
            output_signature=(
                tf.TensorSpec(shape=(self._nside, self._nside, 1), dtype=float),
                tf.TensorSpec(shape=(), dtype=float),
            ),
        )

        train_ds = self._setup_tfds(tfds, 0, train_size)
        val_ds = self._setup_tfds(tfds, train_size, val_size)
        test_ds = self._setup_tfds(tfds, train_size + val_size, test_size)
        return train_ds, test_ds, val_ds
