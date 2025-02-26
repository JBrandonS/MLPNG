import logging
import h5py
import numpy as np
import healpy as hp
import tensorflow as tf
from tensorflow.data.experimental import assert_cardinality
from itertools import product

logger = logging.getLogger(__name__)


class ALMDataset:

    file_path: str
    duplicates: int
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    start_idx: int
    end_idx: int
    shapes: list[str]
    l_str: str

    @classmethod
    def fromCore(
        cls,
        core,
        shapes=["all"],
        start_idx=0,
        end_idx: int | None = None,
        lensed=False,
    ):
        return cls(
            file_path=core.file,
            x_shape=core.alm_shape[2:],
            duplicates=core.ndups,
            shapes=shapes,
            start_idx=start_idx,
            end_idx=end_idx,
            lensed=lensed,
            fnl_min=core.fnl_min,
            fnl_max=core.fnl_max,
        )

    def __init__(
        self,
        file_path,
        x_shape,
        duplicates=10,
        shapes=None,
        start_idx=0,
        end_idx: int | None = None,
        lensed=False,
        seed=None,
        x_dtype=tf.float32,
        y_dtype=tf.float32,
        fnl_min=-1000.0,
        fnl_max=1000.0,
    ):
        # setup shapes
        if isinstance(shapes, str):
            shapes = [shapes]
        if shapes is None or "all" in shapes:
            shapes = ["local", "equilateral", "orthogonal"]

        self.shapes = shapes
        self.y_shape = (len(shapes),)
        self.duplicates = duplicates
        self.l_str = "lensed" if lensed else "unlensed"
        self.file_path = file_path
        self.start_idx = start_idx
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.fnl_min = fnl_min
        self.fnl_max = fnl_max
        if end_idx is None:
            with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
                self.end_idx = f["alm_l"].shape[0]
        else:
            self.end_idx = end_idx
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.end_idx - self.start_idx

    def _get(self, sim, dup):
        with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
            x = f["alm_l"][sim]
            y = self.rng.uniform(
                low=self.fnl_min,
                high=self.fnl_max,
                size=len(self.shapes),
            )

            for i, shape in enumerate(self.shapes):
                alm_nl = f[self.l_str][shape]["alm_nl"][sim]
                x = x + y[i, None] * alm_nl
        return x, y

    def _load(self, i):
        x, y = tf.py_function(
            func=self._get,
            inp=[i[0], i[1]],
            Tout=(self.x_dtype, self.y_dtype),
            name="py_loader",
        )  # type: ignore
        x.set_shape(self.x_shape)
        y.set_shape(self.y_shape)
        return x, y

    def split(
        self,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        to_tf=True,
        batch_size=32,
    ):
        # split the data into train, val, test
        n = self.end_idx - self.start_idx
        train_end = self.start_idx + int(n * train_size)
        val_end = train_end + int(n * val_size)
        test_end = np.min((val_end + int(n * test_size), self.end_idx))

        cls = self.__class__  # so we dont have to change this if we subclass
        kwargs = {
            "file_path": self.file_path,
            "duplicates": self.duplicates,
            "x_shape": self.x_shape,
            "x_dtype": self.x_dtype,
            "y_dtype": self.y_dtype,
            "shapes": self.shapes,
            "lensed": "lensed" in self.l_str,
            "fnl_min": self.fnl_min,
            "fnl_max": self.fnl_max,
        }
        train = cls(**kwargs, start_idx=self.start_idx, end_idx=train_end)
        val = cls(**kwargs, start_idx=train_end, end_idx=val_end)
        test = cls(**kwargs, start_idx=val_end, end_idx=test_end)

        if to_tf:
            train = train.to_tf(batch_size, True)
            val = val.to_tf(batch_size, True)
            test = test.to_tf(batch_size, False)
        return train, val, test

    def to_tf(self, batch_size=32, reshuffle=True):
        indices = list(
            product(range(self.start_idx, self.end_idx), range(self.duplicates))
        )

        ds = tf.data.Dataset.from_generator(
            lambda: indices,
            output_signature=tf.TensorSpec(shape=(2,), dtype=tf.int32),
            name="index_generator",
        )

        ds = ds.map(
            self._load,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
            name="data_loader",
        )

        ds = ds.apply(assert_cardinality(len(indices)))
        ds = ds.cache()
        ds = ds.shuffle(
            buffer_size=min(1024, len(indices)), reshuffle_each_iteration=reshuffle
        )
        ds = ds.batch(
            batch_size,
            # drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
            name="batcher",
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    def get_fisher(self, shape, lensed=False):
        with h5py.File(self.file_path, mode="r", swmr=True, locking=False) as f:
            fisher = f[self.l_str][shape]["fisher"][0]
        return fisher


class MapDataset(ALMDataset):
    @classmethod
    def fromCore(
        cls,
        core,
        shapes=None,
        start_idx=0,
        end_idx: int | None = None,
        lensed=False,
    ):
        return cls(
            file_path=core.file,
            duplicates=core.ndups,
            x_shape=(core.npix, core.npols),
            x_dtype=core.r_dtype,
            y_dtype=core.r_dtype,
            shapes=shapes,
            start_idx=start_idx,
            end_idx=end_idx,
            lensed=lensed,
            nside=core.nside,
            fnl_min=core.fnl_min,
            fnl_max=core.fnl_max,
        )

    def __init__(
        self,
        file_path,
        x_shape,
        duplicates=25,
        x_dtype=tf.complex64,
        y_dtype=tf.float32,
        shapes=None,
        start_idx=0,
        end_idx: int | None = None,
        lensed=False,
        nside=32,
        fnl_min=-1000.0,
        fnl_max=1000.0,
    ):
        super().__init__(
            file_path=file_path,
            x_shape=x_shape,
            duplicates=duplicates,
            shapes=shapes,
            start_idx=start_idx,
            end_idx=end_idx,
            lensed=lensed,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            fnl_min=fnl_min,
            fnl_max=fnl_max,
        )
        self.nside = nside

    def _get(self, sim, dup):
        x, y = super()._get(sim, dup)

        maps = []
        for i in range(x.shape[0]):
            maps.append(hp.alm2map(x[i].astype(complex), nside=self.nside, pol=False))
        maps = np.transpose(maps, (1, 0))
        return maps, y
