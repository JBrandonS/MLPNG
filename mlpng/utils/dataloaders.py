import os
import logging
import healpy as hp

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import TensorSpec
from tensorflow.data import Dataset
from tensorflow.data.experimental import assert_cardinality

logger = logging.getLogger(__name__)


def rotate_ds(inputs, y):
    @tf.py_function(Tout=tf.float32)  # type: ignore
    def rotate_map(inputs):
        lat_angle = tf.random.uniform([1], -90.0, 90.0)  # type: ignore
        lon_angle = tf.random.uniform([1], -180.0, 180.0)  # type: ignore
        rotator = hp.Rotator(rot=[lat_angle, lon_angle], deg=True, inv=True)
        x = hp.reorder(inputs, n2r=True)

        # I have had issues in the past with the pixel weights not being downloaded due to server errors
        # disable here if you get an error with the file not being found on astropy's servers
        x = rotator.rotate_map_alms(x, use_pixel_weights=False)
        return hp.reorder(x, r2n=True)

    x = tf.map_fn(rotate_map, inputs)
    x.set_shape(inputs.shape)  # type: ignore
    return x, y


# @tf.function(jit_compile=True)
def tfds_from_hdf5(
    core,
    x_name="/map",
    y_name="/fnl",
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    data_fraction=1.0,
    shuffle=True,
    reshuffle=True,
    batch_size=64,
    buffer=None,
    cache_prefix="",
):
    # hackish spec TODO
    npix = hp.nside2npix(core.nside)
    x_spec = TensorSpec(shape=[core.ndups, core.npols, npix], dtype=core.r_dtype)
    if y_name == "/fnl":
        y_spec = TensorSpec(shape=[core.ndups, 1, 1], dtype=core.r_dtype)
    else:
        y_spec = TensorSpec(shape=[core.ndups, core.npols, npix], dtype=core.r_dtype)

    ds_x: Dataset = tfio.IODataset.from_hdf5(core.file, x_name, x_spec)
    ds_y: Dataset = tfio.IODataset.from_hdf5(core.file, y_name, y_spec)

    dataset = tf.data.Dataset.zip(ds_x, ds_y)
    dataset = dataset.apply(assert_cardinality(core.nsims * core.narray))

    # Split into train/val/test sets
    train, val, test = split_ds(
        dataset,
        train_size * data_fraction,
        val_size * data_fraction,
        test_size * data_fraction,
    )

    train = process_ds(
        train,
        batch_size,
        buffer=buffer,
        cache_file_suffix=f"{core.name}/{cache_prefix}train",
        shuffle=shuffle,
        reshuffle=reshuffle,
    )
    val = process_ds(
        val,
        batch_size,
        buffer=buffer,
        cache_file_suffix=f"{core.name}/{cache_prefix}val",
        shuffle=shuffle,
        reshuffle=reshuffle,
    )
    test = process_ds(
        test,
        batch_size,
        buffer=buffer,
        cache_file_suffix=f"{core.name}/{cache_prefix}test",
        shuffle=False,
    )
    return train, val, test


def split_ds(ds, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    ds_len = ds.cardinality().numpy()
    train_len = int(ds_len * train_frac)
    val_len = int(ds_len * val_frac)
    test_len = int(ds_len * test_frac)
    test_start = train_len + val_len

    logger.debug(
        "ds_len %s, total split len %s, train %s, val %s, test %s",
        ds_len,
        train_len + val_len + test_len,
        train_len,
        val_len,
        test_len,
    )

    train = ds.take(train_len).apply(assert_cardinality(train_len))
    val = ds.skip(train_len).take(val_len).apply(assert_cardinality(val_len))
    test = ds.skip(test_start).take(test_len).apply(assert_cardinality(test_len))
    return train, val, test


def process_ds(
    ds,
    batch_size=64,
    cache=True,
    unique_cache=True,
    cache_file_suffix=None,
    shuffle=True,
    reshuffle=True,
    buffer=None,
    prefetch=True,
):
    # remove the ndups axis
    ds = ds.unbatch()

    if cache:
        if cache_file_suffix is not None:
            if unique_cache:
                cache_file = f"{os.environ.get('SCRATCH')}/tfcache/{os.environ.get('SLURM_JOB_ID')}/{cache_file_suffix}"
            else:
                cache_file = f"{os.environ.get('SCRATCH')}/tfcache/{cache_file_suffix}"
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            logger.debug(f"Caching to {cache_file}")
            ds = ds.cache(cache_file)
        else:
            logger.debug("Caching to memory")
            ds = ds.cache()

    if shuffle:
        if buffer is None:
            buffer = min(batch_size * 10, ds.cardinality().numpy())
        ds = ds.shuffle(buffer_size=buffer, reshuffle_each_iteration=reshuffle)

    ds = ds.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # if rotate:
    #     ds = ds.map(
    #         lambda x, y: rotate_ds(x, y),
    #         # num_parallel_calls=tf.data.AUTOTUNE,
    #     )

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
