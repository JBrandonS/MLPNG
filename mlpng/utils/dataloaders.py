import os
import io
import copy
import logging
import itertools
import numpy as np
import healpy as hp
import lenspyx
import tensorflow as tf
from joblib import Parallel, delayed
from contextlib import redirect_stdout

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
        fnl_seed=42,
        x_shape=None,
        x_dtype=tf.complex64,
        y_shape=None,
        y_dtype=tf.float32,
        parallel_prefer="processes",
    ):
        # this is shit, since only needed for maps, think about this more
        # set during split when called from mapping
        self.rotate = False
        self.gaussian_mask = False
        self.parallel_prefer = parallel_prefer

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
        self.fnl_seed = fnl_seed

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
            prefer=self.parallel_prefer,
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

        np.random.seed(self.fnl_seed + indices[0])
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
        with Parallel(n_jobs, pre_dispatch="n_jobs", prefer=self.parallel_prefer) as p:
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

        np.random.seed(self.fnl_seed + indices[0])
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
        with Parallel(n_jobs, pre_dispatch="n_jobs", prefer=self.parallel_prefer) as p:
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
        npix = hp.nside2npix(core.nside * 2)

        y_shape = kwargs.pop("y_shape", (None, npix, core.npols))
        y_dtype = kwargs.pop("y_dtype", tf.float32)
        return super().fromCore(core, y_shape=y_shape, y_dtype=y_dtype, **kwargs)

    @staticmethod
    def _alm_to_map_batch(
        alm_lens, alm_phi, rotate, nside, batch_size, shapes, duplicates
    ):
        lens_maps = []
        phi_maps = []
        for batch, dup, shape in itertools.product(
            range(batch_size), range(duplicates), range(len(shapes))
        ):
            if rotate:  # VERY slow
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

            m_lens = hp.alm2map(alm_lens_rotated, nside, pol=False, inplace=True)

            # could this be a problem?
            m_phi = hp.alm2map(alm_phi_rotated, nside * 2, pol=False, inplace=True)

            lens_maps.append(hp.reorder(m_lens, r2n=True))
            phi_maps.append(hp.reorder(m_phi, r2n=True))

        return np.array(lens_maps)[..., None], np.array(phi_maps)[..., None]

    def _generate(self, indices, duplicates):
        batch_size = len(indices)

        np.random.seed(self.fnl_seed + indices[0])
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
        # alm_lens_flat = alm_lens.reshape(-1, alm_lens.shape[-1])

        return self._alm_to_map_batch(
            alm_lens,
            alm_phi,
            self.rotate,
            self.nside,
            batch_size,
            self.shapes,
            duplicates,
        )


class KappaDataset(MapDataset):
    """
    Dataset for generating lensed CMB maps with flexible x/y outputs.

    Generates lensed maps by applying lensing potential phi_scale*alm_phi to unlensed ALMs.
    Supports flexible x (input) and y (output) configurations: any combination of maps (lensed, unlensed, phi, kappa) and parameters (fnl, phi_scale, kappa_scale).

    Example: x_output="lensed", y_output=("unlensed", "phi", "phi_scale") for a de-lensing model that takes lensed map and outputs unlensed map, phi map, and phi_scale.
    """

    @classmethod
    def fromCore(
        cls,
        core: "Core",
        phi_scale: float | tuple[float, float] | list[float] = 1.0,
        kappa_scale: float = 1.0,
        x_output: str | tuple[str, ...] = "lensed",
        y_output: str | tuple[str, ...] = ("fnl",),
        gaussian_mask_prob: float = 0.1,
        lmax_buffer: int | None = None,
        parallel_prefer: str = "processes",
        **kwargs,
    ) -> "KappaDataset":
        """
        Create dataset from Core object.

        Args:
            core: Core configuration object
            phi_scale: Phi-scale configuration (default: 1.0)
                     - Single float: fixed value
                     - Array-like: uniformly sample from array values
                     - Tuple of 2 values: range (min, max) to uniformly sample
            kappa_scale: Scale factor to multiply kappa maps (default: 1.0)
            x_output: Single map/param type for input x (default: "lensed")
                     Options: "lensed", "unlensed", "phi", "kappa", "fnl", "phi_scale"
            y_output: Tuple of map/param types for output y (default: ("fnl",))
                     Options: "lensed", "unlensed", "phi", "kappa", "fnl", "phi_scale"
            gaussian_mask_prob: Probability to zero fnl and phi independently (default: 0.1)
            lmax_buffer: Buffer for lmax in lenspyx operations (default: core.lmax_buffer)
            parallel_prefer: Preference for joblib parallelization (default: "processes").
                           Options: "processes" (default, more stable), "threads" (faster but breaks Jupyter notebooks under some conditions).
        """
        # Ensure y_output is a tuple
        if isinstance(y_output, str):
            y_output = (y_output,)

        shapes = kwargs.pop("shapes", core.shapes)
        nside = kwargs.pop("nside", core.nside)

        # Determine output shapes based on output types
        x_shape = cls._compute_output_shape(
            x_output, core.npix, core.npols, len(core.shapes)
        )
        y_shape = cls._compute_output_shape(
            y_output, core.npix, core.npols, len(core.shapes)
        )

        x_dtype = kwargs.pop("x_dtype", tf.float32)
        y_dtype = kwargs.pop("y_dtype", tf.float32)

        if lmax_buffer is None:
            lmax_buffer = getattr(core, "lmax_buffer", 128)

        return cls(
            file_path=kwargs.pop("file_path", core.file),
            nside=nside,
            shapes=shapes,
            fnl_min=kwargs.pop("fnl_min", core.fnl_min),
            fnl_max=kwargs.pop("fnl_max", core.fnl_max),
            phi_scale=phi_scale,
            kappa_scale=kappa_scale,
            x_output=x_output,
            y_output=y_output,
            gaussian_mask_prob=gaussian_mask_prob,
            fnl_seed=kwargs.pop("fnl_seed", 42),
            lmax_buffer=lmax_buffer,
            x_shape=x_shape,
            x_dtype=x_dtype,
            y_shape=y_shape,
            y_dtype=y_dtype,
            parallel_prefer=parallel_prefer,
            **kwargs,
        )

    @staticmethod
    def _compute_output_shape(
        output_types: str | tuple[str, ...],
        npix: int,
        npols: int,
        nshapes: int,
    ) -> tuple[int | None, ...]:
        """Compute output shape based on output types.

        Note: phi and kappa maps are generated at nside*2 resolution (4x npix),
        while lensed and unlensed maps use the standard nside resolution (npix).
        """
        if isinstance(output_types, str):
            output_types = (output_types,)

        n_maps = 0
        n_params = 0
        has_hires = False  # phi/kappa use 4x resolution
        has_lowres = False  # lensed/unlensed use standard resolution

        for out_type in output_types:
            if out_type in ["phi", "kappa"]:
                n_maps += 1
                has_hires = True
            elif out_type in ["lensed", "unlensed"]:
                n_maps += 1
                has_lowres = True
            elif out_type == "fnl":
                n_params += nshapes
            elif out_type == "phi_scale":
                n_params += 1

        # If single map and no params, keep 3D shape
        if n_maps == 1 and n_params == 0:
            if has_hires:
                # phi/kappa at nside*2: 4x npix
                return (None, 4 * npix, npols)
            else:
                # lensed/unlensed at standard nside: npix
                return (None, npix, npols)
        # If single param and no params, keep 1D shape (1,)
        elif n_params == 1 and n_maps == 0:
            return (None, 1)
        # Multiple items: flatten to 1D
        else:
            # Cannot mix high-res and low-res maps - user must separate them
            if has_hires and has_lowres:
                raise ValueError(
                    "Cannot mix phi/kappa (high-res) with lensed/unlensed (low-res) maps. "
                    "Phi and kappa are generated at nside*2 resolution (4x pixels). "
                    "Use separate output specifications for different resolutions."
                )
            total_size = n_maps * (4 * npix if has_hires else npix) * npols + n_params
            return (None, total_size)

    @staticmethod
    def _process_phi_scale(
        phi_scale: float | tuple[float, float] | list[float],
    ) -> dict[str, float | np.ndarray]:
        """
        Process phi_scale parameter.

        Args:
            phi_scale: Single value, array, or tuple range
                - float: fixed value
                - array-like: uniformly sample from these values
                - tuple of 2 values: (min, max) range to uniformly sample

        Returns:
            dict with 'type' and configuration for sampling
        """
        if isinstance(phi_scale, (tuple, list)) and len(phi_scale) == 2:
            # Range: (min, max)
            return {"type": "range", "min": phi_scale[0], "max": phi_scale[1]}
        elif isinstance(phi_scale, (list, np.ndarray)):
            # Array of values
            return {"type": "array", "values": np.asarray(phi_scale)}
        else:
            # Single value
            return {"type": "fixed", "value": float(phi_scale)}

    def __init__(
        self,
        file_path: str,
        nside: int,
        shapes: list[str] | str | None = None,
        fnl_min: float = -1000.0,
        fnl_max: float = 1000.0,
        phi_scale: float | tuple[float, float] | list[float] = 1.0,
        kappa_scale: float = 1.0,
        x_output: str | tuple[str, ...] = "lensed",
        y_output: str | tuple[str, ...] = ("fnl",),
        gaussian_mask_prob: float = 0.1,
        fnl_seed: int = 42,
        lmax_buffer: int = 128,
        x_shape: tuple[int | None, ...] | None = None,
        x_dtype=tf.float32,
        y_shape: tuple[int | None, ...] | None = None,
        y_dtype=tf.float32,
        parallel_prefer: str = "processes",
        **kwargs,
    ) -> None:
        # Ensure outputs are tuples
        if isinstance(x_output, str):
            x_output = (x_output,)
        if isinstance(y_output, str):
            y_output = (y_output,)

        # Process phi_scale: can be single value, array, or tuple range
        self.phi_scale_config = self._process_phi_scale(phi_scale)
        self.kappa_scale = kappa_scale
        self.x_output = x_output
        self.y_output = y_output
        self.gaussian_mask_prob = gaussian_mask_prob
        self.lmax_buffer = lmax_buffer
        self.core = kwargs.pop("core", None)
        self.gaussian_mask = False  # Handled separately with gaussian_mask_prob

        if shapes is None:
            shapes = ["local"]
        elif isinstance(shapes, str):
            shapes = [shapes]

        if y_shape is None:
            n_params = sum(
                1 if o == "phi_scale" else (len(shapes) if o == "fnl" else 0)
                for o in y_output
                if o in ["fnl", "phi_scale"]
            )
            y_shape = (None, max(1, n_params))

        super().__init__(
            file_path=file_path,
            nside=nside,
            shapes=shapes,
            fnl_min=fnl_min,
            fnl_max=fnl_max,
            fnl_seed=fnl_seed,
            x_shape=x_shape,
            x_dtype=x_dtype,
            y_shape=y_shape,
            y_dtype=y_dtype,
            parallel_prefer=parallel_prefer,
            **kwargs,
        )

        self._setup_lensing_geometry()

    def _setup_lensing_geometry(self) -> None:
        """Pre-compute lenspyx geometry for lensing operations."""
        self.geom_info = ("healpix", {"nside": self.nside})
        self.geom = lenspyx.get_geom(self.geom_info)
        self.lmax = 3 * self.nside - 1
        self.lmax_len = self.lmax + self.lmax_buffer
        fl = np.sqrt(np.arange(self.lmax_len + 1) * np.arange(1, self.lmax_len + 2))
        self.fl = fl
        # kappa coefficients: kappa_lm = -1/2 * ell(ell+1) * phi_lm
        kappa_coeff = (
            -0.5 * np.arange(self.lmax_len + 1) * np.arange(1, self.lmax_len + 2)
        )
        self.kappa_coeff = kappa_coeff

    def _generate_parameters(
        self, indices: np.ndarray, duplicates: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate random fnl and phi parameters with independent seeding and masking."""
        batch_size = len(indices)
        nshapes = len(self.shapes)

        np.random.seed(self.fnl_seed + indices[0])
        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(batch_size, duplicates, nshapes),
        )

        # Generate phi values based on configuration
        phis = np.zeros((batch_size, 1), dtype=np.float32)

        if self.phi_scale_config["type"] == "fixed":
            # Fixed value
            phis[:] = self.phi_scale_config["value"]
        elif self.phi_scale_config["type"] == "range":
            # Uniformly sample from range
            phis = np.random.uniform(
                self.phi_scale_config["min"],
                self.phi_scale_config["max"],
                size=(batch_size, 1),
            )
        elif self.phi_scale_config["type"] == "array":
            # Uniformly sample from array
            phis = np.random.choice(
                self.phi_scale_config["values"], size=(batch_size, 1)
            )

        # Apply Gaussian masking vectorized: independently zero fnl and phi with gaussian_mask_prob
        if self.gaussian_mask_prob > 0:
            fnl_mask = (
                np.random.rand(batch_size, duplicates, 1) < self.gaussian_mask_prob
            )
            phi_mask = np.random.rand(batch_size, 1) < self.gaussian_mask_prob
            fnls[fnl_mask] = 0
            phis[phi_mask] = 0

        return fnls, phis

    def _lens_alms(
        self,
        indices: np.ndarray,
        duplicates: int,
        fnls: np.ndarray,
        phis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load unlensed ALMs and apply lensing with phi_scale-dependent potential."""
        batch_size = len(indices)

        alm_l = get_data(self.file_path, f"alm_l/unlensed", indices)
        alm_nl = np.array(
            [
                get_data(self.file_path, f"alm_nl/unlensed/{s}", indices)
                for s in self.shapes
            ]
        )

        # Load alm_phi from data file
        alm_phi_all = get_data(self.file_path, "alm_phi", indices)

        alms_lensed = np.zeros(
            (batch_size, duplicates, alm_l.shape[1], alm_l.shape[2]),
            dtype=np.complex128,
        )

        with redirect_stdout(io.StringIO()):
            for sim_idx in range(batch_size):
                for dup_idx in range(duplicates):
                    fnl_vals = fnls[sim_idx, dup_idx]
                    alm = np.asarray(alm_l[sim_idx], dtype=np.complex128) + np.einsum(
                        "i...,i...->...", fnl_vals, alm_nl[:, sim_idx]
                    )
                    alm = np.asarray(alm, dtype=np.complex128)

                    phi_scale = phis[sim_idx]
                    alm_phi = alm_phi_all[sim_idx] * phi_scale

                    dlm = hp.almxfl(alm_phi, self.fl)

                    alm_t = np.asarray(alm[0], dtype=np.complex128)

                    lenmap = lenspyx.alm2lenmap(
                        alm_t,
                        dlm,
                        geometry=self.geom_info,
                        nthreads=1,
                    )
                    lenmap = np.ascontiguousarray(lenmap)

                    alm_lensed_t = self.geom.map2alm(
                        lenmap,
                        self.lmax,
                        self.lmax,
                        nthreads=1,
                    )

                    alm_lensed_t = np.asarray(alm_lensed_t, dtype=np.complex128)

                    if alm.shape[0] == 1:
                        alms_lensed[sim_idx, dup_idx, 0] = alm_lensed_t
                    else:
                        alms_lensed[sim_idx, dup_idx, 0, : len(alm_lensed_t)] = (
                            alm_lensed_t
                        )
                        alms_lensed[sim_idx, dup_idx, 1:, :] = alm[1:, :]

        return alms_lensed, alm_phi_all

    def _generate(
        self, indices: np.ndarray, duplicates: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a batch of CMB maps with flexible x/y outputs. Only generates maps when requested."""
        batch_size = len(indices)

        fnls, phis = self._generate_parameters(indices, duplicates)

        # Determine which maps are needed
        all_outputs = set(self.x_output) | set(self.y_output)
        need_lensed = "lensed" in all_outputs
        need_unlensed = "unlensed" in all_outputs
        need_phi = "phi" in all_outputs
        need_kappa = "kappa" in all_outputs

        n_cpus = len(os.sched_getaffinity(0))
        n_jobs = min(n_cpus, batch_size * duplicates)

        maps_dict = {}

        # Only generate lensed maps if needed
        if need_lensed:
            alms_lensed, alm_phi_all = self._lens_alms(indices, duplicates, fnls, phis)
            with Parallel(
                n_jobs, pre_dispatch="n_jobs", prefer=self.parallel_prefer
            ) as p:
                maps_lensed = p(
                    delayed(self._alm_to_map)(alm, self.rotate, self.nside)
                    for batch in alms_lensed
                    for alm in batch
                )
            maps_lensed = np.array(maps_lensed)
            maps_lensed = np.transpose(maps_lensed, (0, 2, 1))
            maps_dict["lensed"] = maps_lensed
        else:
            # Still need alm_phi_all if generating phi or kappa
            if need_phi or need_kappa:
                alm_phi_all = get_data(self.file_path, "alm_phi", indices)

        # Only generate unlensed maps if needed
        if need_unlensed:
            alm_l = get_data(self.file_path, f"alm_l/{self.l_str}", indices)
            alm_nl = np.array(
                [
                    get_data(self.file_path, f"alm_nl/{self.l_str}/{s}", indices)
                    for s in self.shapes
                ]
            )
            alms_unlensed = np.zeros(
                (batch_size, duplicates, alm_l.shape[1], alm_l.shape[2]),
                dtype=np.complex128,
            )
            for sim_idx in range(batch_size):
                for dup_idx in range(duplicates):
                    fnl_vals = fnls[sim_idx, dup_idx]
                    alm = np.asarray(alm_l[sim_idx], dtype=np.complex128) + np.einsum(
                        "i,i...->...", fnl_vals, alm_nl[:, sim_idx]
                    )
                    alm = np.asarray(alm, dtype=np.complex128)
                    alms_unlensed[sim_idx, dup_idx] = alm
                    if alm.shape[0] > 1:
                        alms_unlensed[sim_idx, dup_idx, 1:] = alm[1:]

            with Parallel(
                n_jobs, pre_dispatch="n_jobs", prefer=self.parallel_prefer
            ) as p:
                maps_unlensed = p(
                    delayed(self._alm_to_map)(alm, self.rotate, self.nside)
                    for batch in alms_unlensed
                    for alm in batch
                )
            maps_unlensed = np.array(maps_unlensed)
            maps_unlensed = np.transpose(maps_unlensed, (0, 2, 1))
            maps_dict["unlensed"] = maps_unlensed

        # Only generate phi maps if needed
        if need_phi:
            if "alm_phi_all" not in locals():
                alm_phi_all = get_data(self.file_path, "alm_phi", indices)
            alm_phi_scaled = np.asarray(alm_phi_all * phis, dtype=np.complex128)
            with Parallel(
                n_jobs, pre_dispatch="n_jobs", prefer=self.parallel_prefer
            ) as p:
                maps_phi = p(
                    delayed(self._alm_to_map)(alm, self.rotate, self.nside * 2)
                    for alm in alm_phi_scaled
                )
            maps_phi = np.array(maps_phi)
            maps_phi = np.expand_dims(maps_phi, axis=-1)
            maps_dict["phi"] = maps_phi

        # Only generate kappa (convergence) maps if needed
        if need_kappa:
            if "alm_phi_all" not in locals():
                alm_phi_all = get_data(self.file_path, "alm_phi", indices)
            alm_phi_scaled = np.asarray(alm_phi_all * phis, dtype=np.complex128)
            alm_kappa = np.array(
                [
                    hp.almxfl(
                        np.asarray(alm, dtype=np.complex128),
                        self.kappa_coeff,
                    )
                    for alm in alm_phi_scaled
                ],
                dtype=np.complex128,
            )
            with Parallel(
                n_jobs, pre_dispatch="n_jobs", prefer=self.parallel_prefer
            ) as p:
                maps_kappa = p(
                    delayed(self._alm_to_map)(alm, self.rotate, self.nside * 2)
                    for alm in alm_kappa
                )
            maps_kappa = np.array(maps_kappa)
            maps_kappa = np.repeat(maps_kappa, duplicates, axis=0)
            maps_kappa = np.expand_dims(maps_kappa, axis=-1)
            # Apply kappa_scale multiplier
            maps_kappa = maps_kappa * self.kappa_scale
            maps_dict["kappa"] = maps_kappa

        # Build x output
        x_items = []
        for out_type in self.x_output:
            if out_type in maps_dict:
                x_items.append(maps_dict[out_type])
            elif out_type == "fnl":
                x_items.append(fnls.reshape(batch_size * duplicates, 1))
            elif out_type == "phi_scale":
                x_items.append(phis)

        # Stack x: single map stays (npix, npols), multiple items flatten
        if len(x_items) == 1:
            item = x_items[0]
            if item.ndim == 3 and item.shape[-1] <= 2:  # Maps stay 3D
                x = item
            else:
                # Reshape to (batch_size, duplicates, -1) then flatten to (batch_size*duplicates, -1)
                x = item.reshape(batch_size, duplicates, -1)
                x = x.reshape(batch_size * duplicates, -1)
        else:
            # Flatten all items and stack
            x_flat = []
            for item in x_items:
                if item.ndim >= 2:
                    x_reshaped = item.reshape(batch_size, duplicates, -1)
                    x_flat.append(x_reshaped.reshape(batch_size * duplicates, -1))
                else:
                    x_flat.append(item)
            x = np.hstack(x_flat)

        # Build y output
        y_items = []
        for out_type in self.y_output:
            if out_type in maps_dict:
                y_items.append(maps_dict[out_type])
            elif out_type == "fnl":
                y_items.append(fnls.reshape(batch_size * duplicates, 1))
            elif out_type == "phi_scale":
                y_items.append(phis)

        # Stack y: single map stays 3D, multiple items flatten
        if len(y_items) == 1:
            item = y_items[0]
            if item.ndim == 3 and item.shape[-1] <= 2:  # Maps stay 3D
                y = item
            else:
                # Reshape to (batch_size, duplicates, -1) then flatten to (batch_size*duplicates, -1)
                y = item.reshape(batch_size, duplicates, -1)
                y = y.reshape(batch_size * duplicates, -1)
        else:
            # Flatten all items and stack
            y_flat = []
            for item in y_items:
                if item.ndim >= 2:
                    y_reshaped = item.reshape(batch_size, duplicates, -1)
                    y_flat.append(y_reshaped.reshape(batch_size * duplicates, -1))
                else:
                    y_flat.append(item)
            y = np.hstack(y_flat)

        return x, y


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

        np.random.seed(self.fnl_seed + indices[0])
        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(len(indices), len(self.shapes), duplicates, 1, 1),  # todo add pols
        )

        alms = Parallel(4, prefer=self.parallel_prefer)(
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

        np.random.seed(self.fnl_seed + indices[0])
        fnls = np.random.uniform(
            low=self.fnl_min,
            high=self.fnl_max,
            size=(batch_size, len(self.shapes), duplicates, 1, 1),  # todo add pols
        )

        n_cpus = len(os.sched_getaffinity(0))  # number of CPUs, accounting for slurm
        alm_jobs = min(n_cpus, batch_size)
        map_jobs = min(n_cpus, duplicates * batch_size)

        alms = Parallel(alm_jobs, prefer=self.parallel_prefer)(
            delayed(get_alm)(idx, fnls[i]) for i, idx in enumerate(indices)
        )

        maps = Parallel(map_jobs, prefer=self.parallel_prefer)(
            delayed(process)(sim) for batches in alms for sim in batches  # type: ignore
        )

        maps = np.transpose(maps, (0, 2, 1))  # type: ignore
        fnls = fnls.transpose(1, 2, 3, 4, 0)
        fnls = np.reshape(fnls, (duplicates * batch_size, len(self.shapes)))

        return maps, fnls
