from re import A
import numpy as np
import h5py
import json

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from healpy.sphtfunc import Alm


class AlmLoader(Sequence):
    def __init__(
        self,
        file_path,
        batch_size=1,
        shuffle=False,
        shuffle_buffer=10,
        seed=None,
        cache=True,
        dtype=np.float64,
        name="AlmLoader"
    ):
        # setup some attrributes
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.cache = cache
        self.dtype = dtype
        self.seed = seed

        # load in the file
        self.file = h5py.File(self.file_path, mode="r", swmr=True, locking=False)

        # get some basic info
        (self._nsims, self._npol, self._ndata) = np.array(self.file["alm"]).shape
        self.length = self._nsims * self._npol
        self.lmax = Alm.getlmax(self._ndata)

        # This is ugly but it gets a tuple from the json string read in from the settings of the config file
        # and then uses it to generate the fnl values, since they are not in the cache file
        fnl_range = self.file["settings"].get("fnl_range", ["[-1, 1]"])[0]
        fnl_range = tuple(json.loads(fnl_range))

        # generate the fnl values
        _rng = np.random.default_rng(self.seed)
        self.fnls = _rng.uniform(fnl_range[0], fnl_range[1], (self._nsims, self._npol))

        # get the shape that our dataset will be in
        self.shape = (self.lmax + 1, self.lmax + 1, 2)

        # create the index map
        self.idx_map = np.fromfunction(
            lambda l, m: Alm.getidx(self.lmax, l, m),
            (self.shape[0], self.shape[1]),
            dtype=np.int32,
        )

        # create the dataset generator
        self._ds = tf.data.Dataset.from_generator(
            lambda: self,  # uses the __getitem__ method
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(), dtype=self.dtype),
            ),
            name=name,
        )

    def __getitem__(self, index):
        # convert the index to a tuple of (i, j) indexing sim and pol
        i, j = np.unravel_index(index, (self._nsims, self._npol))

        # now we can get the data
        # probably the slowest part of the code
        alm = np.array(self.file["alm"][i, j])
        almng = np.array(self.file["almng"][i, j])
        fnl = self.fnls[i, j]

        # combine the alms
        alm_complete = alm + fnl * almng

        # converts data(i) -> data(l, m), also splits the real and imaginary parts and adds the index map
        pos_enc = self.idx_map / self._ndata / 100
        data = np.zeros(self.shape, dtype=self.dtype)
        data[:, :, 0] = np.real(alm_complete[self.idx_map]) + pos_enc
        data[:, :, 1] = np.imag(alm_complete[self.idx_map]) + pos_enc

        return data, fnl

    def __len__(self):
        return self.length

    def _get_dataset(self, start, step):
        data = self._ds.skip(start).take(step)
        data = data.apply(tf.data.experimental.assert_cardinality(step))
        if self.cache:
            data = data.cache()
        if self.shuffle:
            data = data.shuffle(
                self.shuffle_buffer, seed=self.seed, reshuffle_each_iteration=True
            )
        data = data.batch(self.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        return data.prefetch(tf.data.AUTOTUNE)

    def get_split(self, train_frac, test_frac, val_frac=None):
        """
        Splits the dataset into training, testing, and optionally validation sets.

        This method calculates the sizes of the training and testing sets based on the provided fractions,
        and then creates the actual datasets.

        Parameters:
        train_frac (float): The fraction of the dataset to use for training.
        test_frac (float): The fraction of the dataset to use for testing.
        val_frac (float, optional): The fraction of the dataset to use for validation. If None, no validation set is created. Defaults to None.

        Returns:
        tuple: The training and testing datasets, and the validation dataset if `val_frac` is not None.
        """

        n = self.length
        train_size = int(n * train_frac)
        test_size = int(n * test_frac)

        train_ds = self._get_dataset(0, train_size)
        test_ds = self._get_dataset(train_size, test_size)
        if val_frac is not None:
            val_size = int(n * val_frac)
            val_ds = self._get_dataset(train_size + test_size, val_size)

            return train_ds, test_ds, val_ds
        else:

            return train_ds, test_ds
