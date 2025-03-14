import os
import copy
import logging
import h5py
import numpy as np
import healpy as hp
import tensorflow as tf
from joblib import Parallel, delayed
from tensorflow.data.experimental import assert_cardinality

logger = logging.getLogger(__name__)


class ALMDataset:
    @classmethod
    def fromCore(cls, core, **kwargs):
        file_path = kwargs.pop("file_path", core.file)
        alm_shape = kwargs.pop("alm_shape", core.alm_shape[1:])
        fnl_min = kwargs.pop("fnl_min", core.fnl_min)
        fnl_max = kwargs.pop("fnl_max", core.fnl_max)
        alm_dtype = kwargs.pop("alm_dtype", core.c_dtype)
        fnl_dtype = kwargs.pop("fnl_dtype", core.r_dtype)

        return cls(
            file_path=file_path,
            alm_shape=alm_shape,
            fnl_min=fnl_min,
            fnl_max=fnl_max,
            alm_dtype=alm_dtype,
            fnl_dtype=fnl_dtype,
            **kwargs,
        )

    def __init__(
        self,
        file_path,
        shapes=None,
        start_idx=0,
        end_idx: int | None = None,
        lensed=False,
        alm_shape=None,
        fnl_min=-1000.0,
        fnl_max=1000.0,
        alm_dtype=tf.complex64,
        fnl_dtype=tf.float32,
    ):
        # setup shapes
        if isinstance(shapes, str):
            shapes = [shapes]
        if shapes is None or "all" in shapes:
            shapes = ["local", "equilateral", "orthogonal"]

        self.shapes = shapes
        self.fnl_shape = (len(shapes),)
        self.l_str = "lensed" if lensed else "unlensed"
        self.file_path = file_path
        self.start_idx = start_idx
        self.fnl_min = fnl_min
        self.fnl_max = fnl_max
        self.alm_dtype = alm_dtype
        self.fnl_dtype = fnl_dtype

        if end_idx is None:
            with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
                self.end_idx = f["alm_l"]["unlensed"].shape[0]
        else:
            self.end_idx = end_idx

        # get the alm shape, well (pol, nelem)
        if alm_shape is None:
            with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
                self.alm_shape = f["alm_l"]["unlensed"].shape[2:]
        else:
            self.alm_shape = alm_shape

    def __len__(self):
        return self.end_idx - self.start_idx

    def get_fisher(self, shape, lensed=False):
        with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
            fisher = f["fisher"][shape][0]
        return fisher

    def split(
        self,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        to_tf=True,
        cache_dir=None,
        **to_tf_kwargs,
    ):
        """
        Splits the dataset into training, validation, and test sets.

        Parameters:
        -----------
        train_size : float, optional
            Proportion of the dataset to include in the training set (default is 0.8).
        val_size : float, optional
            Proportion of the dataset to include in the validation set (default is 0.1).
        test_size : float, optional
            Proportion of the dataset to include in the test set (default is 0.1).
        to_tf : bool, optional
            Whether to convert the splits to TensorFlow datasets before returning (default is True).
        cache_dir: str, optional
            Base directory for caching the datasets. If not provided, defaults to $SCRATCH/tf_cache.
            You must also provide cache_file for this to have meaning
        **to_tf_kwargs : dict
            Additional keyword arguments to pass to the `to_tf` method. Some important ones are batch_size, cache_file and duplicates.
        Returns:
        --------
        tuple
            A tuple containing the training, validation, and test sets. If `to_tf` is True,
            the sets are returned as TensorFlow datasets; otherwise, they are returned as
            copies of the original dataset with updated indices.

        Notes:
        --------
            If you provide a cache_file argument for to_tf, this should be a base file name i.e. 'my_model' instead of a full file name i.e. 'cache_dir/my_model.cache'.
            this allows us to make 3 cache files for the test, val, and train splits here.

            duplicates can be provided as an int or a list of length 3, if it is an int, all three datasets will have the same number of duplicates.
        """
        n_total = self.end_idx - self.start_idx

        # calculate the end points for the datasets
        train_end = self.start_idx + int(n_total * train_size)
        val_end = train_end + int(n_total * val_size)
        test_end = np.min((val_end + int(n_total * test_size), self.end_idx))

        # each dataset is a copy of the original with updated indices
        train = copy.copy(self)
        train.start_idx = self.start_idx
        train.end_idx = train_end

        val = copy.copy(self)
        val.start_idx = train_end
        val.end_idx = val_end

        test = copy.copy(self)
        test.start_idx = val_end
        test.end_idx = test_end

        logger.debug(
            "Splitting '%s' into train: %d:%d, val: %d:%d, test: %d:%d",
            self.file_path,
            train.start_idx,
            train.end_idx,
            val.start_idx,
            val.end_idx,
            test.start_idx,
            test.end_idx,
        )

        if to_tf:
            # we support duplicates being a int, so all three datasets are the same
            # or a list of length 3, a different number of duplicates for each dataset
            dups = to_tf_kwargs.pop("duplicates", 1)
            if isinstance(dups, int):
                dups = [dups] * 3
            else:
                assert len(dups) == 3, "Duplicates must be an int or a list of length 3"

            # check the cache_file and if provided build up a list of files for each dataset
            # if cache_dir is not provided, we default to $SCRATCH/tf_cache
            # these will do nothing if cache_final_load=False
            cache_file = to_tf_kwargs.pop("cache_file", "")
            if cache_file:
                if cache_dir is None:
                    cache_dir = os.path.join(os.environ["SCRATCH"], "tf_cache")

                # make sure the cache dir exists
                os.makedirs(cache_dir, exist_ok=True)

                # create array of strings for the cache files, one for each dataset
                cache = [
                    os.path.join(cache_dir, f"{cache_file}-{name}.cache")
                    for name in ["train", "val", "test"]
                ]
            else:
                # will be in memory cache
                cache = [""] * 3

            # Finally convert the datasets to tf
            train = train.to_tf(cache_file=cache[0], duplicates=dups[0], **to_tf_kwargs)
            val = val.to_tf(cache_file=cache[1], duplicates=dups[1], **to_tf_kwargs)
            # test always should have reshuffle off since we want to plot the same data
            to_tf_kwargs.pop("reshuffle", None)
            test = test.to_tf(
                reshuffle=False, cache_file=cache[2], duplicates=dups[2], **to_tf_kwargs
            )
        return train, val, test

    def to_tf(
        self,
        batch_size=1,
        reshuffle=True,
        buffer_size=1024,
        duplicates=1,
        cache_file="",
        cache_partial_load=False,
        cache_final_load=True,
        batch_n_calls=tf.data.AUTOTUNE,
        batch_deterministic=False,
        batch_drop_remainder=False,
        prefetch_n_calls=tf.data.AUTOTUNE,
    ):
        if cache_file and cache_final_load:
            logger.debug("Using cache file: %s", cache_file)

        # ds = tf.data.Dataset.from_generator(
        #     self._generator,
        #     output_signature=(
        #         tf.TensorSpec(shape=self.x_shape, self.x_dtype), tf.TensorSpec(shape=self.y_shape, self.y_dtype),
        #     )
        # )

        ds = tf.data.Dataset.range(self.start_idx, self.end_idx)
        ds = ds.map(
            self._partial_load,
            # Dont enable parallel calls, it will cause issues with the hdf5 file
            # this is probably a threading / GIL issue that is planned to be fixed in newer python / tf versions (3.11)
            # num_parallel_calls=tf.data.AUTOTUNE,
        )

        if cache_partial_load:
            # the idea is to prevent reading from the file multiple times for speed, so cache before repeating
            # could possibly cause memory issues as I do not think TF is smart enough to free the memory later
            ds = ds.cache()

        ds = ds.repeat(duplicates)

        ds = ds.map(
            self._finalize,
            # dont enabled parallel calls, it will cause slow downs due to the GIL
            # num_parallel_calls=tf.data.AUTOTUNE,
        )

        if cache_final_load:
            # cache the data after preprocessing, if you cache to a file this can let you skip the preprocessing on future runs
            # if you update the data creation, you will need to remove the cache file to see the changes
            ds = ds.cache(cache_file)

        ds = ds.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=reshuffle,
        )
        ds = ds.batch(
            batch_size,
            num_parallel_calls=batch_n_calls,
            deterministic=batch_deterministic,
            drop_remainder=batch_drop_remainder,
        )
        ds = ds.prefetch(prefetch_n_calls)
        return ds

    def _generate(self, i):
        """
        Generator function to yield data from the dataset.
        """
        for i in range(self.start_idx, self.end_idx):
            alm_l, alm_nls = self._partial_load_py(i)
            yield alm_l, alm_nls
            results = Parallel(n_jobs=-1)(
                delayed(read_data)(file_path, sim, l_str, shapes) for sim in sims
            )

    def _get_gen(self):
        return Parallel(n_jobs=-1, return_as="generator")(
            delayed(read_data)(file_path, sim, l_str, shapes) for sim in sims
        )

    #     return results

    def _partial_load_py(self, sim):
        """
        Partially loads data from an HDF5 file for a given simulation.

        We load the data from our hdf5 files into two tensors, one holding alm_l, and a second holding an array
        which contains an alm_nl value per shape.

        Parameters:
        sim (int): The simulation index to load data for.

        Returns:
        tuple: A tuple containing:
            - alm_l (numpy.ndarray): The loaded alm_l data for the given simulation.
            - alm_nls (list of numpy.ndarray): A list of loaded alm_nl data for each shape for the given simulation.
        """
        with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
            alm_l = f["alm_l"][self.l_str][sim]
            alm_nls = [f["alm_nl"][self.l_str][shape][sim] for shape in self.shapes]
        return alm_l, alm_nls

    @tf.function
    def _partial_load(self, i):
        """
        TensorFlow function to load partial data. This is split from _partial_load_py mostly for inheritance reasons.
        """
        return tf.py_function(
            self._partial_load_py,
            [i],
            [self.alm_dtype, self.alm_dtype],
        )

    @tf.function
    def _finalize(self, x, y):
        """
        Finalize the dataset by applying alm = alm_l + sum_i f_nl_(i) * alm_nl_(i) for each shape i.
        """
        # maybe look into fine tuning with a different distribution for f_nl at high epoch
        fnl = tf.random.uniform(
            shape=[1],
            minval=self.fnl_min,
            maxval=self.fnl_max,
            dtype=self.fnl_dtype,
        )
        # tf wants einsum args to be the same dtype
        fnl_c = tf.cast(fnl, dtype=y.dtype)

        # do the alm = alm_l + sum_i f_nl_(i) * alm_nl_(i)
        alm = x + tf.einsum("i,ijk->jk", fnl_c, y)

        # set the shape information for tf
        alm.set_shape(self.alm_shape)

        return alm, fnl


class MapDataset(ALMDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        # notice this is transposed from core.map_shape
        map_shape = kwargs.pop("map_shape", (core.npix, core.npols))
        map_dtype = kwargs.pop("map_dtype", core.r_dtype)
        nside = kwargs.pop("nside", core.nside)

        return super().fromCore(
            core, map_shape=map_shape, map_dtype=map_dtype, nside=nside, **kwargs
        )

    def __init__(
        self,
        file_path,
        map_shape,
        nside,
        start_idx=1,
        end_idx=1001,
        map_dtype=tf.float32,
        rotate=True,
        **kwargs,
    ):
        self.map_shape = map_shape
        self.map_dtype = map_dtype
        self.nside = nside
        self.rotate = rotate
        super().__init__(
            file_path=file_path, start_idx=start_idx, end_idx=end_idx, **kwargs
        )

    def _finalize_py(self, x, y):
        alm = x.numpy().astype(np.complex128)

        if self.rotate:
            # Apply a random rotation to the maps
            rotator = hp.Rotator(
                deg=True,
                rot=[
                    np.random.uniform(-180, 180),
                    np.random.uniform(-90, 90),
                    np.random.uniform(0, 360),
                ],
            )
            alm = rotator.rotate_alm(alm)

        maps = hp.alm2map(alm, self.nside, pol=False)

        # deepsphere wants nest ordering
        maps = hp.reorder(maps, r2n=True)

        # transpose to get pols as last dim, channels
        maps = np.transpose([maps], (1, 0))
        return maps, y

    @tf.function
    def _finalize(self, x, y):
        alm, fnl = super()._finalize(x, y)

        maps, y = tf.py_function(
            self._finalize_py,
            [alm, fnl],
            [self.map_dtype, self.fnl_dtype],
        )  # type: ignore

        maps.set_shape(self.map_shape)
        y.set_shape(self.fnl_shape)
        return maps, y


class elsnerDataset(ALMDataset):
    def __init__(self, file_path, start_idx=1, end_idx=1001, **kwargs):
        # this function mostly exist to change the default start and end due to elsener starting at 1
        super().__init__(file_path, start_idx=start_idx, end_idx=end_idx, **kwargs)

    def _partial_load_py(self, sim):
        i = str(sim.numpy()).zfill(4)
        alm_l = hp.read_alm(f"data/elsner/alm_l_{i}_v3.fits", hdu=(1))
        alm_nl = hp.read_alm(f"data/elsner/alm_nl_{i}_v3.fits", hdu=(1))

        # we scale from the dimensionless units to microkelvin, only needed for elsner
        alm_l *= 2.725e6
        alm_nl *= 2.725e6

        return [alm_l], [[alm_nl]]


class elsnerMapDataset(elsnerDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        map_shape = kwargs.pop("map_shape", (core.npix, core.npols))
        map_dtype = kwargs.pop("map_dtype", tf.float32)
        nside = kwargs.pop("nside", core.nside)
        return super().fromCore(
            core, map_shape=map_shape, map_dtype=map_dtype, nside=nside, **kwargs
        )

    def __init__(
        self, file_path, map_shape, nside, map_dtype=tf.float32, rotate=True, **kwargs
    ):
        self.map_shape = map_shape
        self.map_dtype = map_dtype
        self.nside = nside
        self.rotate = rotate

        super().__init__(file_path=file_path, **kwargs)

    def _finalize_py(self, x, y):
        alm = x.numpy().astype(complex)

        if self.rotate:
            # Apply a random rotation to the maps
            rotator = hp.Rotator(
                deg=True,
                rot=[
                    np.random.uniform(-180, 180),
                    np.random.uniform(-90, 90),
                    # np.random.uniform(0, 360),
                ],
                # inv=True,
            )
            alm = rotator.rotate_alm(alm)

        maps = hp.alm2map(alm, self.nside, pol=False)
        maps = hp.reorder(maps, r2n=True)
        maps = np.transpose(maps, (1, 0))
        return maps, y

    @tf.function
    def _finalize(self, x, y):
        alm, fnl = super()._finalize(x, y)

        maps, y = tf.py_function(
            self._finalize_py,
            [alm, fnl],
            [self.map_dtype, self.fnl_dtype],
        )  # type: ignore

        maps.set_shape(self.map_shape)
        y.set_shape(self.fnl_shape)
        return maps, y
