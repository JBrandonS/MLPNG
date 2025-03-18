import os
import copy
import logging
from pickle import TRUE
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
        x_shape = kwargs.pop("x_shape", core.alm_shape[1:])
        x_dtype = kwargs.pop("x_dtype", core.c_dtype)
        y_shape = kwargs.pop("y_shape", (None, 1))
        y_dtype = kwargs.pop("y_dtype", core.r_dtype)
        fnl_min = kwargs.pop("fnl_min", core.fnl_min)
        fnl_max = kwargs.pop("fnl_max", core.fnl_max)

        return cls(
            file_path=file_path,
            fnl_min=fnl_min,
            fnl_max=fnl_max,
            x_shape=x_shape,
            x_dtype=x_dtype,
            y_shape=y_shape,
            y_dtype=y_dtype,
            **kwargs,
        )

    def __init__(
        self,
        file_path,
        shapes=None,
        start_idx=0,
        end_idx: int | None = None,
        lensed=False,
        fnl_min=-1000.0,
        fnl_max=1000.0,
        x_shape=None,
        x_dtype=tf.complex64,
        y_shape=None,
        y_dtype=tf.float32,
    ):
        # setup shapes
        if isinstance(shapes, str):
            shapes = [shapes]
        if shapes is None or "all" in shapes:
            shapes = ["local", "equilateral", "orthogonal"]
        self.shapes = shapes
        self.l_str = "lensed" if lensed else "unlensed"
        self.file_path = file_path
        self.fnl_min = fnl_min
        self.fnl_max = fnl_max

        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.y_shape = y_shape if y_shape else (None, len(shapes))
        self.y_dtype = y_dtype

        self.start_idx = start_idx
        if end_idx is None:
            with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
                self.end_idx = f["alm_l"]["unlensed"].shape[0]
        else:
            self.end_idx = end_idx

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
            to_tf_kwargs.pop("shuffle", None)
            test = test.to_tf(
                shuffle=False, cache_file=cache[2], duplicates=dups[2], **to_tf_kwargs
            )
        return train, val, test

    def to_tf(
        self,
        gen_batch_size=32,
        duplicates=1,
        unbatch=True,
        cache=True,
        cache_file="",
        buffer_size=1024,
        shuffle=True,
        reshuffle=True,
        batch_size=1,
        batch_n_calls=tf.data.AUTOTUNE,
        batch_deterministic=False,
        batch_drop_remainder=False,
        prefetch_n_calls=tf.data.AUTOTUNE,
    ):
        if cache and cache_file:
            logger.debug("Using cache file: %s", cache_file)

        ds = tf.data.Dataset.from_generator(
            self._generator,
            args=[gen_batch_size, duplicates],
            output_signature=(
                tf.TensorSpec(shape=self.x_shape, dtype=self.x_dtype),
                tf.TensorSpec(shape=self.y_shape, dtype=self.y_dtype),
            ),
        )

        if unbatch:
            ds = ds.unbatch()

        if cache:
            # cache the data after preprocessing, if you cache to a file this can let you skip the preprocessing on future runs
            # if you update the data creation, you will need to remove the cache file to see the changes
            ds = ds.cache(cache_file)

        if shuffle:
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
        return ds.prefetch(prefetch_n_calls)

    def _generate(self, indices, duplicates, batch_size):
        """
        Generates a single batch of data and fnl values for the given indices.
        """
        with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
            alm_ls = np.array(f["alm_l"][self.l_str][indices])
            alm_nls = np.array(
                [f["alm_nl"][self.l_str][s][indices] for s in self.shapes]
            )

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(self.shapes), duplicates, len(indices), 1),
        )  # shape (len(shapes), dups, batch_size, 1)

        alm_ls = alm_ls[None, ...]
        alm_nls = alm_nls[:, None, ...]

        alm = alm_ls + np.einsum("i...,i...->...", fnls, alm_nls)
        return alm, fnls

    def _generator(self, batch_size, duplicates, n_jobs=2):
        """
        Returns a parallel generator to read data from the HDF5 file. This is used to load the data in parallel.
        You probably dont need to change anything here but instead in _generate"""

        # we make a batch of indices to read from the HDF5 file,
        # these are then split into batch_size batches, with the last batch holding the remainder
        indices = np.arange(self.start_idx, self.end_idx)
        batched_indices = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        # we use parallel to generate the data in parallel, this is done by splitting the indices into batches and then calling _generate on each batch
        return Parallel(
            n_jobs,
            return_as="generator",
            pre_dispatch="n_jobs",
            # timeout=60 * 60 * 8,
        )(
            delayed(self._generate)(idxs, duplicates, batch_size)
            for idxs in batched_indices
        )


class MapDataset(ALMDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        x_shape = kwargs.pop("x_shape", (None, core.npix, core.npols))
        x_dtype = kwargs.pop("x_dtype", tf.float32)
        nside = kwargs.pop("nside", core.nside)
        return super().fromCore(
            core, x_shape=x_shape, x_dtype=x_dtype, nside=nside, **kwargs
        )

    def __init__(self, file_path, nside, rotate=True, **kwargs):
        self.nside = nside
        self.rotate = rotate
        super().__init__(file_path=file_path, **kwargs)

    def _generate(self, indices, duplicates, batch_size):
        def process(alm):
            o_lmax = hp.Alm.getlmax(alm.shape[-1])
            n_lmax = 3 * self.nside - 1
            a = hp.resize_alm(alm, o_lmax, o_lmax, n_lmax, n_lmax)

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
                a = rotator.rotate_alm(a)

            m = hp.alm2map(a, self.nside, pol=False)
            return hp.reorder(m, r2n=True)

        idx_len = len(indices)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(self.shapes), duplicates, idx_len, 1, 1),
        )

        n_cpus = len(os.sched_getaffinity(0))  # number of CPUs, accounting for slurm
        map_jobs = min(n_cpus, duplicates * batch_size)

        with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
            alm_l = np.array([f["alm_l"][self.l_str][indices]])
            alm_nl = np.array(
                [f["alm_nl"][self.l_str][s][indices] for s in self.shapes]
            )

        alms = alm_l + np.einsum("i...,i...->...", fnls, alm_nl)

        maps = Parallel(map_jobs)(
            delayed(process)(sim) for batches in alms for sim in batches
        )

        maps = np.transpose(maps, (0, 2, 1))
        fnls = np.reshape(fnls, (-1, len(self.shapes)))

        return maps, fnls


class elsnerDataset(ALMDataset):
    def __init__(self, file_path, start_idx=1, end_idx=1001, rotate=True, **kwargs):
        self.rotate = rotate
        super().__init__(file_path, start_idx=start_idx, end_idx=end_idx, **kwargs)

    def _generate(self, indices, duplicates, batch_size):
        def process_alm(i, fnl):
            i = str(i).zfill(4)
            alm_l = hp.read_alm(f"data/elsner/alm_l_{i}_v3.fits", hdu=(1))
            alm_nl = hp.read_alm(f"data/elsner/alm_nl_{i}_v3.fits", hdu=(1))

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
                alm_l = rotator.rotate_alm(alm_l)
                alm_nl = rotator.rotate_alm(alm_nl)

            alm_l = np.array([[alm_l]]) * 2.725e6
            alm_nl = np.array([[[alm_nl]]]) * 2.725e6

            alm = alm_l + np.einsum("i...,i...->...", fnl, alm_nl)
            return np.array(alm)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(indices), len(self.shapes), duplicates, 1, 1),  # todo add pols
        )

        alms = Parallel(4)(
            delayed(process_alm)(idx, fnls[i]) for i, idx in enumerate(indices)
        )
        return np.array(alms), fnls


class elsnerMapDataset(elsnerDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        x_shape = kwargs.pop("x_shape", (None, core.npix, core.npols))
        x_dtype = kwargs.pop("x_dtype", tf.float32)
        nside = kwargs.pop("nside", core.nside)
        return super().fromCore(
            core, x_shape=x_shape, x_dtype=x_dtype, nside=nside, **kwargs
        )

    def __init__(self, file_path, nside, **kwargs):
        self.nside = nside
        super().__init__(file_path=file_path, **kwargs)

    def _generate(self, indices, duplicates, batch_size):
        def get_alm(i, fnl):
            i = str(i).zfill(4)
            alm_l = hp.read_alm(f"data/elsner/alm_l_{i}_v3.fits", hdu=(1))
            alm_nl = hp.read_alm(f"data/elsner/alm_nl_{i}_v3.fits", hdu=(1))

            alm_l = np.array([[alm_l]]) * 2.725e6
            alm_nl = np.array([[[alm_nl]]]) * 2.725e6

            return alm_l + np.einsum("i...,i...->...", fnl, alm_nl)

        def process(alm):
            a = alm
            # o_lmax = hp.Alm.getlmax(alm.shape[-1])
            # n_lmax = 3 * self.nside - 1
            # a = hp.resize_alm(alm, o_lmax, o_lmax, n_lmax, n_lmax)

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
                a = rotator.rotate_alm(a)

            m = hp.alm2map(a, self.nside, pol=False)
            return hp.reorder(m, r2n=True)

        idx_len = len(indices)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(idx_len, len(self.shapes), duplicates, 1, 1),  # todo add pols
        )

        n_cpus = len(os.sched_getaffinity(0))  # number of CPUs, accounting for slurm
        alm_jobs = min(n_cpus, idx_len)
        map_jobs = min(n_cpus, duplicates * batch_size)

        alms = Parallel(alm_jobs)(
            delayed(get_alm)(idx, fnls[i]) for i, idx in enumerate(indices)
        )

        maps = Parallel(map_jobs)(
            delayed(process)(sim) for batches in alms for sim in batches
        )

        maps = np.transpose(maps, (0, 2, 1))
        fnls = np.reshape(fnls, (-1, len(self.shapes)))

        return maps, fnls
