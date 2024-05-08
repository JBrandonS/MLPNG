import os
import logging

import h5py
import numpy as np
import tensorflow as tf
from healpy.sphtfunc import Alm
from numpy import unravel_index
from numpy.random import default_rng
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.data.experimental import assert_cardinality
from tensorflow.keras import Input
from tensorflow.keras.utils import Sequence

logger = logging.getLogger(__name__)


class DataLoaderBase(Sequence):
    """
    A base class used to load and manage the datasets.

    Attributes
    ----------
    batch_size : int
        The number of samples per local batch.
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
    """

    def __init__(
        self,
        file_path,
        name=None,
        batch_size=1,
        shuffle=False,
        shuffle_buffer=1000,
        normalize=False,
        cache=False,
        seed=None,
        dtype=np.float32,
        num_replicas="auto",
        init_ds=True,
    ):
        """
        Initializes a DataLoader object.

        Args:
            file_path (str): The path to the file.
            name (str, optional): The name of the DataLoader. Defaults to class name.
            batch_size (int, optional): The batch size. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            shuffle_buffer (int, optional): The buffer size for shuffling. Defaults to 1000.
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
            cache (bool, optional): Whether to cache the data. Defaults to False.
            seed (int, optional): The seed value for randomization. Defaults to None.
            dtype (numpy.dtype, optional): The data type. Defaults to np.float32.
            num_replicas (str or int, optional): The number of replicas. Defaults to "auto".
            init_ds (bool, optional): Whether to initialize the dataset. Defaults to True.
        """
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

        if num_replicas == "auto":
            # tested on superpod but would not be surprised if this doesn't work on other systems
            self.num_replicas = len(tf.config.list_physical_devices("GPU")) or 1
            logger.debug(f"auto detected num_replicas: {self.num_replicas}")
        else:
            # make sure num_replicas is >= 1
            self.num_replicas = int(num_replicas) if int(num_replicas) > 0 else 1

        self._ds = None
        self.length = 0
        self.shape = (None,)

        if init_ds:
            self._init_ds()

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
        assert (
            self._ds is not None
        ), "self._ds is None, did you forget to call _init_ds?"
        data = self._ds.skip(start).take(step)

        # fixes an issue with tf not knowing the size of the dataset
        data = data.apply(assert_cardinality(step))

        if self.normalize:
            # normalizer = tf.keras.layers.Normalization(axis=None)
            # normalizer.adapt(self._ds)
            # data = data.map(normalizer, num_parallel_calls=AUTOTUNE)
            data = data.map(self._normalize, num_parallel_calls=AUTOTUNE)
        if self.cache:
            data = data.cache()
        if self.shuffle:
            buffer_size = step if self.shuffle_buffer is None else self.shuffle_buffer
            data = data.shuffle(buffer_size, self.seed, reshuffle_each_iteration=True)

        data = data.batch(
            # with keras, self.batch_size is local batch size, but we want this to be global batch size
            # see: https://keras.io/guides/distributed_training_with_tensorflow/
            self.batch_size * self.num_replicas,
            drop_remainder=True,  # we dont want any partial batches
            # deterministic=False,  # we dont care about order
            num_parallel_calls=AUTOTUNE,
        )
        return data.prefetch(AUTOTUNE)

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

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError("This method must be implemented in a subclass")

    def _init_ds(self):
        raise NotImplementedError("This method must be implemented in a subclass")

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

    def _get_tfds_filename(self):
        return self.file_path.replace(".hdf5", f".{self.name}.tfds")

    def save_as_tfds(self, tfds_file=None, exists_ok=False, force=False):
        """
        Saves the dataset as a TensorFlow Dataset (tfds) file.
        This can be much fast to load than the python generator method once saved.
        """
        assert self._ds is not None, "self._ds is None, call _init_ds"

        if tfds_file is None:
            tfds_file = self._get_tfds_filename()

        if os.path.exists(tfds_file):
            if exists_ok:
                if not force:
                    logger.info(f"File {tfds_file} already exists, skipping save")
                    return
                else:
                    logger.info("Overwriting existing tfds file")
                    os.remove(tfds_file)
            else:
                raise ValueError(f"File {self.file_path} already exists")

        logger.info(f"Saving {self.file_path} as {tfds_file}")
        if self.normalize:
            self.normalize = False
            logger.debug("Disabling normalization for saving")
            self._ds.save(tfds_file)
            self.normalize = True
        else:
            self._ds.save(tfds_file)

    def as_tfds(self, auto_convert=False):
        """
        Returns the dataset as a TensorFlow Dataset (tfds) object.
        If auto_convert is true, it will convert and save the dataset to tfds if it does not exist.
        """
        tfds_file = self._get_tfds_filename()
        if not os.path.exists(tfds_file):
            if auto_convert:
                logger.info(f"Converting {self.file_path} to tfds")
                self.save_as_tfds()
            else:
                raise FileNotFoundError(f"File {tfds_file} does not exist")

        return TFDSLoader(
            tfds_file,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            normalize=self.normalize,
            cache=self.cache,
            seed=self.seed,
            dtype=self.dtype,
            num_replicas=self.num_replicas,  # type: ignore
        )


class TFDSLoader(DataLoaderBase):
    """
    A class used to load and manage TensorFlow Datasets saved as tfds.
    This assumes all preprocessing has been done and the data is ready to be used.
    """

    def _init_ds(self):
        if self.file_path.endswith(".hdf5"):
            logger.debug(f"Auto Converting file name {self.file_path} to tfds")
            self.file_path = self._get_tfds_filename()

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} does not exist")

        self._ds = Dataset.load(self.file_path)
        self.length = self._ds.cardinality().numpy()
        self.shape = self._ds.element_spec[0].shape

        logger.info(
            f"Loaded {self.file_path} with {self.length} samples, and data shape {self.shape}"
        )

    def save_as_tfds(self, exists_ok=False, force=False):
        raise NotImplementedError("This method is not implemented for TFDSLoader")

    def as_tfds(self, auto_convert=False):
        logger.error("Already a TFDS format")
        return self


class PatchLoader(DataLoaderBase):
    """Loads the patches in as channel last format, shape: (nside, nside, 1)"""

    def _init_ds(self):
        self.file = h5py.File(self.file_path, mode="r", swmr=True, locking=False)
        (
            self._nsims,
            self._npol,
            self._npatches,
            self._nside,
            _,
        ) = self.file["patch"].shape

        self.length = self._nsims * self._npol * self._npatches
        self.shape = (self._nside, self._nside, 1)
        self.patches = self.file["patch"]

        logger.info(
            f"Loading {self.file_path} with {self.length} samples, and data shape {self.shape}"
        )

        self._ds = Dataset.from_generator(
            self.__iter__,
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(1,), dtype=self.dtype),
            ),
            name=self.name,
        )

    def __getitem__(self, index):
        # Convert the flat index to a multidimensional index
        i, j, k = unravel_index(index, (self._nsims, self._npol, self._npatches))

        # we also need to add the channel dimension as TF expects it
        patch = np.array(self.patches[i, j, k])
        patch = np.expand_dims(patch, axis=-1)
        fnl = self.file["fnl"][i, j]
        return patch, fnl


class AlmLoader(DataLoaderBase):
    """
    A class used to load and manage the Alm datasets.
    Converts the raw(flat) alms into a square tensor with the real and imaginary parts interleaved.
    Shape: (lmax, lmax, 2) if channels_last else (2, lmax, lmax)
    """

    def __init__(self, file_path, channels_last=False, **kwargs):
        self.channels_last = channels_last

        # we dont store the file, it will close on gc after the alms and fnls are destroyed
        file = h5py.File(file_path, mode="r", swmr=True, locking=False)
        logger.debug("Data keys: " + ", ".join(file.keys()))

        # this lazy loads the alm and fnl data
        self.alms = file["alm"]
        self.fnls = file["fnl"]
        (self.nsims, self.npol, self.ndata) = np.shape(self.alms)
        self.lmax = Alm.getlmax(self.ndata)

        # setup the idx map and positional encoding
        # this converts the flat alms to a grid of (lmax, lmax)
        # healpy is happy to give bad values if bad input in provided,
        # so we limit the input to the valid range, otherwise set to 0
        self.idx_map = np.fromfunction(
            lambda l, m: np.where(m <= l, Alm.getidx(self.lmax, l, m), 0),
            (self.lmax, self.lmax),
            dtype=np.int32,
        )

        # generate a positional encoding which is just a small value from 0 - 1e-3
        self.pos_enc = self.idx_map / self.ndata * 1e-3

        super().__init__(file_path, **kwargs)

    def _init_ds(self):
        # get the shape and length that our dataset will be in
        self.length = self.nsims * self.npol
        if self.channels_last:
            self.shape = (self.lmax, self.lmax, 2)
        else:
            self.shape = (2, self.lmax, self.lmax)

        logger.info(
            f"Loading {self.file_path} with {self.length} samples, and data shape {self.shape}"
        )

        # now create our dataset from generator
        # I am not sure if this is the best way to get the dataset, and it might be causing some slowdowns
        # TODO: Look into
        self._ds = Dataset.from_generator(
            self.__iter__,
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(1,), dtype=self.dtype),
            ),
            name=self.name,
        )

    def __getitem__(self, index):
        # convert the index to a tuple of (i, j) indexing sim and pol
        i, j = np.unravel_index(index, (self.nsims, self.npol))

        # now we can get the data and label
        alm = np.array(self.alms[i, j])[self.idx_map]
        fnl = self.fnls[i, j]

        # converts data(i) -> data(l, m), also splits the real and imaginary parts and adds the positional encoding
        data = np.empty(self.shape, dtype=self.dtype)
        if self.channels_last:
            data[..., 0] = np.real(alm) + self.pos_enc
            data[..., 1] = np.imag(alm) + self.pos_enc
        else:
            data[0, ...] = np.real(alm) + self.pos_enc
            data[1, ...] = np.imag(alm) + self.pos_enc
        return data, fnl

    def _get_tfds_filename(self):
        # save to diffrent files depending on the channels_last setting
        arg_str = "cl" if self.channels_last else "cf"
        return self.file_path.replace(".hdf5", f".{self.name}.{arg_str}.tfds")


class AlmLoaderV2(AlmLoader):
    """Interleaves the real/complex instead of adding a new dim
    Shape: (lmax, 2*lmax) if channels_last else (2*lmax, lmax)
    """

    def _init_ds(self):
        self.length = self.nsims * self.npol
        if self.channels_last:
            self.shape = (self.lmax, 2 * self.lmax)
        else:
            self.shape = (2 * self.lmax, self.lmax)

        # now create our dataset from generator
        self._ds = Dataset.from_generator(
            self.__iter__,
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(1,), dtype=self.dtype),
            ),
            name=self.name,
        )

    def __getitem__(self, index):
        # convert the index to a tuple of (i, j) indexing sim and pol
        i, j = np.unravel_index(index, (self.nsims, self.npol))

        # now we can get the data and label
        alm = np.array(self.alms[i, j])
        fnl = self.fnls[i, j]

        # converts data(i) -> data(l, m), add the positional encoding and interleaves the data
        data = np.empty(self.shape, dtype=self.dtype)
        if self.channels_last:
            data[..., 0::2] = np.real(alm[self.idx_map]) + self.pos_enc
            data[..., 1::2] = np.imag(alm[self.idx_map]) + self.pos_enc
        else:
            data[0::2] = np.real(alm[self.idx_map]) + self.pos_enc
            data[1::2] = np.imag(alm[self.idx_map]) + self.pos_enc
        return data, fnl


class AlmLoaderV2_concat(AlmLoaderV2):
    """Concatenates the real and complex parts instead of interleaving them
    Shape: (lmax, 2*lmax) if channels_last else (2*lmax, lmax)
    """

    def __getitem__(self, index):
        # convert the index to a tuple of (i, j) indexing sim and pol
        i, j = np.unravel_index(index, (self.nsims, self.npol))

        # now we can get the data and label
        alm = np.array(self.alms[i, j])
        fnl = self.fnls[i, j]

        # converts data(i) -> data(l, m), also splits the real and imaginary parts and adds the index map
        data = np.concatenate(
            (
                np.real(alm[self.idx_map]) + self.pos_enc,
                np.imag(alm[self.idx_map]) + self.pos_enc,
            ),
            axis=-1 if self.channels_last else 0,
        )
        return data, fnl


class AlmLoaderRaw(AlmLoader):
    """Returns just the raw alm data split into real and complex parts
    Shape: (ndata, 2) if channels_last is true else (2, ndata)"""

    def _init_ds(self):
        self.length = self.nsims * self.npol
        if self.channels_last:
            self.shape = (self.ndata, 2)
        else:
            self.shape = (2, self.ndata)

        logger.info(
            f"Loading {self.file_path} with {self.length} samples, and data shape {self.shape}"
        )

        # now create our dataset from generator
        self._ds = Dataset.from_generator(
            self.__iter__,
            output_signature=(
                tf.TensorSpec(shape=self.shape, dtype=self.dtype),
                tf.TensorSpec(shape=(1,), dtype=self.dtype),
            ),
            name=self.name,
        )

    def __getitem__(self, index):
        i, j = np.unravel_index(index, (self.nsims, self.npol))

        # now we can get the data and label
        alm = np.array(self.alms[i, j])
        fnl = self.fnls[i, j]

        data = np.empty(self.shape, dtype=self.dtype)
        if self.channels_last:
            data[..., 0] = np.real(alm)
            data[..., 1] = np.imag(alm)
        else:
            data[0] = np.real(alm)
            data[1] = np.imag(alm)
        return data, fnl
