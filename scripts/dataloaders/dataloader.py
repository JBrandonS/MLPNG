import numpy as np
import h5py

import tensorflow as tf
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self, file_name, shuffle=True, shuffle_buffer=100, seed=None, normalize=False, batch_size=1, cache=True):
        self.file_name = file_name
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.normalize = normalize
        self.batch_size = batch_size
        self.cache = cache
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

        self.length = self._nsims * self._ndup * self._npol * self._npatches
        self.idxs = np.arange(self.length)

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

    def _setup_tfds(self, ds, start, step):
        data = ds.skip(start).take(step)
        data = data.apply(tf.data.experimental.assert_cardinality(step))
        if self.cache:
            data = data.cache()
        if self.shuffle:
            data = data.shuffle(self.shuffle_buffer, seed=self.seed)
        data = data.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        return data.prefetch(tf.data.AUTOTUNE)

    def get_split(
        self, train_frac=0.8, test_frac=0.1, val_frac=0.1
    ):
        n = self.length
        train_size = int(n * train_frac)
        test_size = int(n * test_frac)

        ds = tf.data.Dataset.from_generator(
            lambda: self,
            output_signature=(
                tf.TensorSpec(shape=(self._nside, self._nside, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )

        train_ds = self._setup_tfds(ds, 0, train_size)
        test_ds = self._setup_tfds(ds, train_size, test_size)

        if val_frac is not None:
            val_size = int(n * val_frac)
            val_ds = self._setup_tfds(ds, train_size + test_size, val_size)
            return train_ds, test_ds, val_ds
        else:
            return train_ds, test_ds
