import json
import math

import h5py
import numpy as np
import tensorflow as tf

from numpy import unravel_index
from numpy.random import default_rng

from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.data.experimental import assert_cardinality
from tensorflow.keras.utils import Sequence

from healpy.sphtfunc import Alm


class PatchLoader(Sequence):
    """
    A class used to load and manage our generated patches.

    Attributes
    ----------
    batch_size : int
        The number of samples per batch.
    shuffle : bool
        Whether to shuffle the data.
    shuffle_buffer : int | None
        The number of samples to use when shuffling the data.
        The whole dataset is used if None.
    cache : bool
        Whether to cache the data in memory.
    dtype : data-type
        The desired data-type for the arrays.
    seed : int
        Random seed for shuffling the data.
    shape : tuple
        The shape of the output data.
    length : int
        The total number of samples per batch.
    normalize : bool
        Whether to normalize the data.
    _ds : tf.data.Dataset
        The TensorFlow Dataset object.

    Methods
    -------
    __getitem__(index)
        Returns the data and label for the sample at the given index.
    _normalize(data, label)
        Normalizes the data.
    __len__()
        Returns the length of batches.
    _init_ds()
        Initializes the dataset object, length, and shape.
    _get_dataset(start, step)
        Returns a dataset starting from 'start' and containing 'step' samples.
    get_split(train_frac, test_frac, val_frac=None)
        Splits the dataset into training, testing, and optionally validation sets.
    """

    def __init__(
        self,
        file_path,
        name="PatchLoader",
        batch_size=16,
        shuffle=False,
        shuffle_buffer=1000,
        normalize=False,
        cache=True,
        seed=None,
        dtype=np.float32,
        dataset=None,
        shape=(None,),
        length=0,
        num_replicas="auto",
    ):
        # setup some attrributes
        self.file_path = file_path
        self.name = name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.cache = cache
        self.normalize = normalize
        self.seed = seed if seed is not None else default_rng().integers(0, 2**32 - 1)
        self.dtype = dtype
        self.shape = shape
        self.length = length

        if num_replicas == "auto":
            # tested on superpod but would not be suprised if this doesnt work on other systems
            self.num_replicas = len(tf.config.list_physical_devices("GPU")) or 1
        else:
            # make sure num_replicas is >= 1
            self.num_replicas = int(num_replicas) if int(num_replicas) > 1 else 1

        if dataset is None:
            self._init_ds()
        else:
            self._ds = dataset

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __getitem__(self, index):
        # Convert the flat index to a multidimensional index
        i, j, k, l = unravel_index(
            index, (self._nsims, self._ndup, self._npol, self._npatches)
        )

        # we also need to add the channel dimension as TF expects it
        patch = np.array(self.file["patches"][i, j, k, l][:, :, None])
        fnl = np.array(self.file["fnls"][i, j])
        return patch, fnl

    def _init_ds(self):
        if hasattr(self, "_ds"):
            raise ValueError("Dataset already initialized")

        self.file = file = h5py.File(self.file_path, mode="r", swmr=True, locking=False)
        (
            self._nsims,
            self._ndup,
            self._npol,
            self._npatches,
            self._nside,
            _,
        ) = file["patches"].shape

        self.length = self._nsims * self._ndup * self._npol * self._npatches
        self.shape = (self._nside, self._nside, 1)

        self._ds = Dataset.from_generator(
            self.__iter__,
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(), dtype=self.dtype),
            ),
            name=self.name,
        )

        opts = tf.data.Options()
        opts.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        opts.deterministic = False  # we dont need things returned in order
        self._ds = self._ds.with_options(opts)

    @tf.function
    def _normalize(self, data, label):
        """
        Normalizes the values to be between 0 and 1.
        """
        min_val = tf.reduce_min(data)
        max_val = tf.reduce_max(data)
        data = (data - min_val) / (max_val - min_val)
        return data, label

    def _get_dataset(self, start, step):
        """
        Creates a subset of the dataset starting from the 'start' index and taking 'step' number of elements.

        The subset is processed according to the class's settings: it is normalized if 'self.normalize' is True,
        cached if 'self.cache' is True, and shuffled if 'self.shuffle' is True. The subset is then batched with
        'self.batch_size' number of elements per batch.

        Parameters:
        start (int): The index to start the subset from.
        step (int): The number of elements to include in the subset.

        Returns:
        tf.data.Dataset: The processed subset of the dataset, ready for training or evaluation.
        """
        data = self._ds.skip(start).take(step)
        data = data.apply(assert_cardinality(step))
        if self.normalize:
            data = data.map(self._normalize, num_parallel_calls=AUTOTUNE)
        if self.cache:
            data = data.cache()
        if self.shuffle:
            buffer_size = step if self.shuffle_buffer is None else self.shuffle_buffer
            data = data.shuffle(buffer_size, self.seed, reshuffle_each_iteration=True)

        # multiply by the number of gpus to get the correct batch size for distributed systems
        data = data.batch(
            self.batch_size * self.num_replicas,
            drop_remainder=True,
            num_parallel_calls=AUTOTUNE,
        )
        return data.prefetch(AUTOTUNE)

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
        train_size = int(self.length * train_frac)
        test_size = int(self.length * test_frac)

        train_ds = self._get_dataset(0, train_size)
        test_ds = self._get_dataset(train_size, test_size)

        if val_frac is not None:
            val_size = int(self.length * val_frac)
            val_ds = self._get_dataset(train_size + test_size, val_size)
            return train_ds, test_ds, val_ds

        return train_ds, test_ds


class AlmLoader(PatchLoader):
    def __init__(
        self,
        file_path,
        name="AlmLoader",
        **kwargs,
    ):
        super().__init__(file_path, name, **kwargs)

    def _init_ds(self):
        if hasattr(self, "_ds"):
            raise ValueError("Dataset already initialized")

        self._file = h5py.File(self.file_path, mode="r", swmr=True, locking=False)

        (self._nsims, self._npol, self._ndata) = self._file["alm"].shape

        # get the shape that our dataset will be in
        lmax = Alm.getlmax(self._ndata)
        self.shape = (lmax + 1, lmax + 1, 2)
        self.length = self._nsims * self._npol

        # Loads in the fnl from the data
        # This will be in a dataset=array(bytestring) format, [b'[-100, 100]'], so grab the first element and decode it
        # convert this to a tuple using json to get our fnl range
        fnl_range_dataset = self._file["settings"]["fnl_range"]
        fnl_range_json = fnl_range_dataset[0].decode()
        fnl_range = tuple(json.loads(fnl_range_json))

        # generate the fnl values
        self.fnls = default_rng(self.seed).uniform(
            fnl_range[0], fnl_range[1] + 1, (self._nsims, self._npol)
        )

        # create the index map for our transformation
        self.idx_map = np.fromfunction(
            lambda l, m: Alm.getidx(lmax, l, m),
            (self.shape[0], self.shape[1]),
            dtype=np.int32,
        )

        # create the position encoding
        self.pos_enc = self.idx_map / self._ndata / 100

        # now create our dataset from generator
        self._ds = Dataset.from_generator(
            self.__iter__,
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(), dtype=self.dtype),
            ),
            name=self.name,
        )

    def __getitem__(self, index):
        # convert the index to a tuple of (i, j) indexing sim and pol
        i, j = np.unravel_index(index, (self._nsims, self._npol))

        # now we can get the data
        # probably the slowest part of the code
        alm = np.array(self._file["alm"][i, j])
        almng = np.array(self._file["almng"][i, j])
        fnl = self.fnls[i, j]

        # combine the alms
        alm_complete = alm + fnl * almng

        # converts data(i) -> data(l, m), also splits the real and imaginary parts and adds the index map
        data = np.zeros(self.shape, dtype=self.dtype)
        data[:, :, 0] = np.real(alm_complete[self.idx_map]) + self.pos_enc
        data[:, :, 1] = np.imag(alm_complete[self.idx_map]) + self.pos_enc

        # # trying sparse tensor
        # # Find the indices where the tensor is not zero
        # indices = tf.where(tf.not_equal(data, 0))
        # # Gather the non-zero values
        # values = tf.gather_nd(data, indices)
        # # Get the shape of the original tensor
        # shape = tf.shape(data, out_type=tf.int64)
        # # Create the sparse tensor
        # data = tf.SparseTensor(indices, values, shape)

        return data, fnl


class TFDSLoader(PatchLoader):
    """
    A class used to load and manage TensorFlow Datasets saved as tfds.
    This assumes all preprocessing has been done and the data is ready to be used.
    """

    def __init__(self, file_path, name="TFDSLoader", **kwargs):
        super().__init__(file_path, name, **kwargs)

    def _init_ds(self):
        if hasattr(self, "_ds"):
            raise ValueError("Dataset already initialized")

        self._ds = Dataset.load(self.file_path)
        self.length = self._ds.cardinality().numpy()
        self.shape = self._ds.element_spec[0].shape

    def __getitem__(self, index):
        return self._ds.skip(index).take(1).as_numpy_iterator().next()
