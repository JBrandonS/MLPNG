import os
import math
import copy
import logging
import h5py
import itertools
import numpy as np
import healpy as hp
import tensorflow as tf
from joblib import Parallel, delayed

from mlpng.utils import get_data

logger = logging.getLogger(__name__)


class ALMDataset:
    @classmethod
    def fromCore(cls, core, **kwargs):
        # values can be overridden by providing them in kwargs
        # otherwise will be taken from the core object
        shapes = kwargs.pop("shapes", core.shapes)
        return cls(
            file_path=kwargs.pop("file_path", core.file),
            shapes=shapes,
            fnl_min=kwargs.pop("fnl_min", core.fnl_min),
            fnl_max=kwargs.pop("fnl_max", core.fnl_max),
            x_shape=kwargs.pop("x_shape", core.alm_shape[1:]),
            x_dtype=kwargs.pop("x_dtype", core.c_dtype),
            y_shape=kwargs.pop("y_shape", (None, len(shapes))),
            y_dtype=kwargs.pop("y_dtype", core.r_dtype),
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
        # this is shit, since only needed for maps, think about this more
        # set during split when called from mapping
        self.rotate = False
        self.gaussian_mask = False

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
            self.end_idx = get_data(file_path, f"alm_l/{self.l_str}").shape[0]
        else:
            self.end_idx = end_idx  # type: ignore

        # lets print some debug info
        if logger.isEnabledFor(logging.DEBUG):
            fisher_mat = get_data(file_path, f"fisher_matrix/{self.l_str}", 0)
            marg_likes = get_data(file_path, f"marginal_likelihoods/{self.l_str}", 0)

            logger.debug("Error information from '%s'...", file_path)
            logger.debug("Fisher Matrix: %s", fisher_mat)
            logger.debug("Marginal Likelihoods: %s", marg_likes)

            for s in shapes:
                fisher = get_data(file_path, f"fisher/{self.l_str}/{s}", 0)
                std_div = 1 / np.sqrt(fisher)
                logger.debug("Fisher for shape %s: %s, std div: %s", s, fisher, std_div)

    def __len__(self):
        return self.end_idx - self.start_idx

    def split(
        self,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        to_tf=True,
        cache_dir=None,
        rotate=[False, False, False],
        gaussian_mask=[False, False, False],
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
        train.rotate = rotate[0]
        train.gaussian_mask = gaussian_mask[0]

        val = copy.copy(self)
        val.start_idx = train_end
        val.end_idx = val_end
        val.rotate = rotate[1]
        val.gaussian_mask = gaussian_mask[1]

        test = copy.copy(self)
        test.start_idx = val_end
        test.end_idx = test_end
        test.rotate = rotate[2]
        test.gaussian_mask = gaussian_mask[2]

        logger.debug(
            "Splitting '%s' into train: %d:%d (%d), val: %d:%d (%d), test: %d:%d (%d)",
            self.file_path,
            train.start_idx,
            train.end_idx,
            len(train),
            val.start_idx,
            val.end_idx,
            len(val),
            test.start_idx,
            test.end_idx,
            len(test),
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

            # test always should have shuffle off since we want to plot the same data
            to_tf_kwargs.pop("shuffle", None)
            test = test.to_tf(
                shuffle=False, cache_file=cache[2], duplicates=dups[2], **to_tf_kwargs
            )
        return train, val, test

    def to_tf(
        self,
        gen_batch_size=8,
        duplicates=1,
        unbatch=True,
        cache=True,
        cache_file="",
        shuffle=True,
        buffer_size=128,
        reshuffle=True,
        batch_size=64,
        batch_n_calls=tf.data.AUTOTUNE,
        batch_deterministic=False,
        batch_drop_remainder=False,
        prefetch_n_calls=tf.data.AUTOTUNE,
    ):
        """
        Converts a data generator into a TensorFlow `tf.data.Dataset` pipeline with various preprocessing options.
        Args:
            gen_batch_size (int, optional): The batch size for the data generator, these batches will be processed in python and sent as one to the tf pipeline.
                This setting should be used to optimize the CPU load vs memory transfer, probably want it on the order of the number of CPUS. Defaults to 32.
            duplicates (int, optional): Number of samples to generate per 'seed' batch. Defaults to 1.
            unbatch (bool, optional): Whether to unbatch the dataset, splitting batched samples into individual samples. Defaults to True.
            cache (bool, optional): Whether to cache the dataset. Defaults to True.
            cache_file (str, optional): Path to a file for caching the dataset. If empty, caching is done in memory. Defaults to "".
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            buffer_size (int, optional): Buffer size for shuffling the dataset. Defaults to 1024.
            reshuffle (bool, optional): Whether to reshuffle the dataset on each iteration. Defaults to True.
            batch_size (int, optional): The batch size for the final dataset. Defaults to 1.
            batch_n_calls (int, optional): Number of parallel calls for batching. Defaults to `tf.data.AUTOTUNE`.
            batch_deterministic (bool, optional): Whether batching should be deterministic. Defaults to False.
            batch_drop_remainder (bool, optional): Whether to drop the last batch if it has fewer than `batch_size` elements. Defaults to False.
            prefetch_n_calls (int, optional): Number of parallel calls for prefetching. Defaults to `tf.data.AUTOTUNE`.
        Returns:
            tf.data.Dataset: A TensorFlow dataset pipeline with the specified preprocessing steps applied.
        Notes:
            - If `cache` is enabled and `cache_file` is provided, the dataset will be cached to the specified file.
                If the cache file exists, it will be reused unless deleted manually.
        """

        if cache and cache_file:
            logger.debug("Using cache file: '%s'", cache_file)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        ds = tf.data.Dataset.from_generator(
            self._generator,
            args=[gen_batch_size, duplicates],
            output_signature=(
                tf.TensorSpec(shape=self.x_shape, dtype=self.x_dtype),  # type: ignore
                tf.TensorSpec(shape=self.y_shape, dtype=self.y_dtype),  # type: ignore
            ),
        )

        if unbatch:
            # this takes us from 1 * (batch, ...) samples to batch * (....)
            # i.e. a single (batch * duplicates, pols, data) tensor -> (batch * duplicates) tensors of shape (pols, data)
            ds = ds.unbatch()

        ds_len = len(self) * duplicates
        ds = ds.apply(tf.data.experimental.assert_cardinality(ds_len))

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

    def _generate(self, indices, duplicates):
        """
        Generates a single batch of data, returning the data and fnl values.
        """
        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(self.shapes), duplicates, len(indices), 1, 1),
        )

        alm_l = get_data(self.file_path, f"alm_l/{self.l_str}", indices)
        alm_nl = np.array(
            [
                get_data(self.file_path, f"alm_nl/{self.l_str}/{s}", indices)
                for s in self.shapes
            ]
        )

        alms = alm_l + np.einsum("i...,i...->...", fnls, alm_nl)
        return alms, fnls

    def _generator(
        self,
        batch_size,
        duplicates,
        n_jobs=len(os.sched_getaffinity(0)),
        pre_dispatch="n_jobs",
    ):
        """
        Returns a parallel generator to read data from the file. This is used to load the data in parallel.
        You probably dont need to change anything here but instead in _generate
        """

        # we make a batch of indices to read from the HDF5 file,
        # these are then split into batch_size batches, with the last batch holding the remainder
        # note: temping to shuffle here but h5py wants the arguments sorted too
        # we also shuffle on the TF side so not a big deal
        indices = np.arange(self.start_idx, self.end_idx)
        batched_indices = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        # we use parallel to generate the data in parallel, this is done by splitting the indices into batches and then calling _generate on each batch
        temp_folder = os.environ.get("SCRATCH", None)
        return Parallel(
            min(n_jobs, len(batched_indices)),
            return_as="generator",
            pre_dispatch=pre_dispatch,
            prefer="threads",
            temp_folder=temp_folder,
        )(delayed(self._generate)(i, duplicates) for i in batched_indices)


class MapDataset(ALMDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        x_shape = kwargs.pop("x_shape", (None, core.npix, core.npols))
        x_dtype = kwargs.pop("x_dtype", tf.float32)
        nside = kwargs.pop("nside", core.nside)
        return super().fromCore(
            core, x_shape=x_shape, x_dtype=x_dtype, nside=nside, **kwargs
        )

    def __init__(self, file_path, nside, rotate=True, gaussian_mask=True, **kwargs):
        self.nside = nside
        self.rotate = rotate
        self.gaussian_mask = gaussian_mask
        super().__init__(file_path=file_path, **kwargs)

    @staticmethod
    def _alm_to_map(alm, rotate, nside):
        if rotate:
            # Apply a random rotation to the maps
            rotator = hp.Rotator(
                deg=True,
                rot=[
                    np.random.uniform(-180, 180),
                    np.random.uniform(-90, 90),
                    np.random.uniform(0, 360),
                ],
            )
            alm_rotated = rotator.rotate_alm(alm)
        else:
            alm_rotated = alm

        m = hp.alm2map(alm_rotated, nside, pol=False)
        return hp.reorder(m, r2n=True)

    @staticmethod
    def _mask_fnls(fnls, mask_prob=0.1):  # 0.035859):
        # note mask prob should be a comb factor for all shapes
        # generate a mask along the shapes, dup, sim dimensions
        mask = np.random.rand(*fnls.shape[0:3]) < mask_prob
        fnls[mask] = 0
        return fnls

    def _generate(self, indices, duplicates):
        batch_size = len(indices)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(self.shapes), duplicates, batch_size, 1, 1),
        )

        if self.gaussian_mask:
            fnls = self._mask_fnls(fnls)

        alm_l = get_data(self.file_path, f"alm_l/{self.l_str}", indices)
        alm_nl = np.array(
            [
                get_data(self.file_path, f"alm_nl/{self.l_str}/{s}", indices)
                for s in self.shapes
            ]
        )

        alms = alm_l + np.einsum("i...,i...->...", fnls, alm_nl)

        # number of CPUs, accounting for slurm, use 1 less to avoid overloading
        n_cpus = len(os.sched_getaffinity(0))
        n_jobs = min(n_cpus, duplicates * batch_size)
        with Parallel(n_jobs, pre_dispatch="n_jobs", prefer="threads") as p:
            maps = p(
                delayed(self._alm_to_map)(sim, self.rotate, self.nside)
                for batches in alms
                for sim in batches
            )

        maps = np.transpose(np.array(maps), (0, 2, 1))
        fnls = fnls.transpose(1, 2, 3, 4, 0)
        fnls = np.reshape(fnls, (duplicates * batch_size, len(self.shapes)))
        return maps, fnls

    def split(
        self, rotate=[False, False, False], gaussian_mask=[True, False, False], **kwargs
    ):
        return super().split(rotate=rotate, gaussian_mask=gaussian_mask, **kwargs)


class UnlensMapDataset(MapDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        y_shape = kwargs.pop("y_shape", (None, core.npix, core.npols))
        y_dtype = kwargs.pop("y_dtype", tf.float32)
        return super().fromCore(core, y_shape=y_shape, y_dtype=y_dtype, **kwargs)

    @staticmethod
    def _alm_to_map(alm_lens, alm_unlens, rotate, nside):
        if rotate:
            # Apply a random rotation to the maps
            rotator = hp.Rotator(
                deg=True,
                rot=[
                    np.random.uniform(-180, 180),
                    np.random.uniform(-90, 90),
                    np.random.uniform(0, 360),
                ],
            )
            alm_lens_rotated = rotator.rotate_alm(alm_lens)
            alm_unlensed_rotated = rotator.rotate_alm(alm_unlens)
        else:
            alm_lens_rotated = alm_lens
            alm_unlensed_rotated = alm_unlens

        m_lens = hp.alm2map(alm_lens_rotated, nside, pol=False)
        m_unlensed = hp.alm2map(alm_unlensed_rotated, nside, pol=False)
        return hp.reorder(m_lens, r2n=True), hp.reorder(m_unlensed, r2n=True)

    def _generate(self, indices, duplicates):
        batch_size = len(indices)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(self.shapes), duplicates, batch_size, 1, 1),
        )

        if self.gaussian_mask:
            fnls = self._mask_fnls(fnls)

        alm_l_lens = get_data(self.file_path, f"alm_l/lensed", indices)
        alm_l_unlensed = get_data(self.file_path, f"alm_l/unlensed", indices)

        alm_nl_lens = np.array(
            [
                get_data(self.file_path, f"alm_nl/lensed/{s}", indices)
                for s in self.shapes
            ]
        )
        alm_nl_unlensed = np.array(
            [
                get_data(self.file_path, f"alm_nl/unlensed/{s}", indices)
                for s in self.shapes
            ]
        )

        alm_lens = alm_l_lens + np.einsum("i...,i...->...", fnls, alm_nl_lens)
        alm_unlensed = alm_l_unlensed + np.einsum(
            "i...,i...->...", fnls, alm_nl_unlensed
        )

        # number of CPUs, accounting for slurm
        n_cpus = len(os.sched_getaffinity(0))
        n_jobs = min(n_cpus, duplicates * batch_size)
        with Parallel(n_jobs, pre_dispatch="n_jobs", prefer="threads") as p:
            maps = p(
                delayed(self._alm_to_map)(
                    sim_lens, sim_unlensed, self.rotate, self.nside
                )
                for lens_batches, unlensed_batches in zip(alm_lens, alm_unlensed)
                for sim_lens, sim_unlensed in zip(lens_batches, unlensed_batches)
            )

        map_lens, map_unlensed = zip(*maps)
        map_lens = np.transpose(np.array(map_lens), (0, 2, 1))
        map_unlensed = np.transpose(np.array(map_unlensed), (0, 2, 1))

        return map_lens, map_unlensed


class PhiMapDataset(UnlensMapDataset):
    @classmethod
    def fromCore(cls, core, **kwargs):
        lmax = core.lmax + core.lmax_buffer
        nside = (lmax - 1) // 3
        npix = hp.nside2npix(nside)
        npix = hp.nside2npix(128)  # HAX

        y_shape = kwargs.pop("y_shape", (None, npix, core.npols))
        y_dtype = kwargs.pop("y_dtype", tf.float32)
        return super().fromCore(core, y_shape=y_shape, y_dtype=y_dtype, **kwargs)

    # @staticmethod
    # def _alm_to_map(alm_lens, alm_phi, rotate, nside):
    #     if rotate:
    #         # Apply a random rotation to the maps
    #         rotator = hp.Rotator(
    #             deg=True,
    #             rot=[
    #                 np.random.uniform(-180, 180),
    #                 np.random.uniform(-90, 90),
    #                 np.random.uniform(0, 360),
    #             ],
    #         )
    #         alm_lens_rotated = rotator.rotate_alm(alm_lens)
    #         alm_phi_rotated = rotator.rotate_alm(alm_phi)
    #     else:
    #         alm_lens_rotated = alm_lens
    #         alm_phi_rotated = alm_phi

    #     m_lens = hp.alm2map(alm_lens_rotated, nside, pol=False)
    #     m_phi = hp.alm2map(alm_phi_rotated, nside, pol=False)
    #     return hp.reorder(m_lens, r2n=True), hp.reorder(m_phi, r2n=True)

    @staticmethod
    def _alm_to_map_batch(
        alm_lens, alm_phi, rotate, nside, batch_size, shapes, duplicates
    ):
        # Replicate alm_phi to match the flattened alm_lens structure
        # alm_phi needs to be repeated for each (shape, duplicate) combination per batch item
        alm_phi_expanded = np.repeat(alm_phi, duplicates * len(shapes), axis=0)
        alm_phi_expanded = alm_phi_expanded.reshape(duplicates, len(shapes), -1)

        lens_maps = []
        phi_maps = []
        for batch, dup, shape in itertools.product(
            range(batch_size), range(duplicates), range(len(shapes))
        ):
            if rotate:
                rotator = hp.Rotator(
                    deg=True,
                    rot=[
                        np.random.uniform(-180, 180),
                        np.random.uniform(-90, 90),
                        np.random.uniform(0, 360),
                    ],
                )
                alm_lens_rotated = rotator.rotate_alm(alm_lens[dup, batch, shape])
                alm_phi_rotated = rotator.rotate_alm(alm_phi[batch, shape])
            else:
                alm_lens_rotated = alm_lens[dup, batch, shape]
                alm_phi_rotated = alm_phi[batch]

            # m_lmax = hp.Alm.getlmax(alm_lens_rotated.shape[-1])
            # phi_lmax = hp.Alm.getlmax(alm_phi_rotated.shape[-1])
            # nside_phi = (phi_lmax - 1) // 3

            # 106 -> 128
            nside_phi = 128
            # tf.print(
            #     f"Calculated nside: {nside_phi} from phi_lmax: {phi_lmax}, shape: {alm_phi_rotated.shape}"
            # )

            m_lens = hp.alm2map(alm_lens_rotated, nside, pol=False, inplace=True)
            m_phi = hp.alm2map(alm_phi_rotated, nside_phi, pol=False, inplace=True)

            lens_maps.append(hp.reorder(m_lens, r2n=True))
            phi_maps.append(hp.reorder(m_phi, r2n=True))

        return np.array(lens_maps)[..., None], np.array(phi_maps)[..., None]

    def _generate(self, indices, duplicates):
        batch_size = len(indices)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(self.shapes), duplicates, batch_size, 1, 1),
        )

        if self.gaussian_mask:
            fnls = self._mask_fnls(fnls)

        alm_l_lens = get_data(self.file_path, f"alm_l/lensed", indices)
        alm_nl_lens = np.array(
            [
                get_data(self.file_path, f"alm_nl/lensed/{s}", indices)
                for s in self.shapes
            ]
        )
        alm_lens = alm_l_lens + np.einsum("i...,i...->...", fnls, alm_nl_lens)

        alm_phi = get_data(self.file_path, f"alm_phi", indices)
        alm_phi = alm_phi.astype(np.complex128)  # needed for rotate alm

        # Reshape to flatten duplicates, batch, and shapes into one dimension
        # Original shape: (duplicates, batch_size, shapes, alm_size)
        # Target shape: (duplicates * batch_size * shapes, alm_size)
        alm_lens_flat = alm_lens.reshape(-1, alm_lens.shape[-1])

        return self._alm_to_map_batch(
            alm_lens,
            alm_phi,
            self.rotate,
            self.nside,
            batch_size,
            self.shapes,
            duplicates,
        )


class elsnerDataset(ALMDataset):
    def __init__(self, file_path, start_idx=1, end_idx=1001, rotate=True, **kwargs):
        self.rotate = rotate
        super().__init__(file_path, start_idx=start_idx, end_idx=end_idx, **kwargs)

    def _generate(self, indices, duplicates):
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

    def _generate(self, indices, duplicates):
        def get_alm(i, fnl):
            idx = str(i).zfill(4)
            alm_l = hp.read_alm(f"data/elsner/alm_l_{idx}_v3.fits", hdu=(1))
            alm_nl = hp.read_alm(f"data/elsner/alm_nl_{idx}_v3.fits", hdu=(1))

            alm_l = np.array([[alm_l]]) * 2.725e6
            alm_nl = np.array([[[alm_nl]]]) * 2.725e6

            return alm_l + np.einsum("i...,i...->...", fnl, alm_nl)

        def process(alm):
            a = np.copy(alm)  # prevent modification of the alm in global state

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

        batch_size = len(indices)

        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(batch_size, len(self.shapes), duplicates, 1, 1),  # todo add pols
        )

        n_cpus = len(os.sched_getaffinity(0))  # number of CPUs, accounting for slurm
        alm_jobs = min(n_cpus, batch_size)
        map_jobs = min(n_cpus, duplicates * batch_size)

        alms = Parallel(alm_jobs)(
            delayed(get_alm)(idx, fnls[i]) for i, idx in enumerate(indices)
        )

        maps = Parallel(map_jobs)(
            delayed(process)(sim) for batches in alms for sim in batches  # type: ignore
        )

        maps = np.transpose(maps, (0, 2, 1))  # type: ignore
        fnls = fnls.transpose(1, 2, 3, 4, 0)
        fnls = np.reshape(fnls, (duplicates * batch_size, len(self.shapes)))

        return maps, fnls
