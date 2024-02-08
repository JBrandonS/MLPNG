import sys
from numpy.random import uniform

from scripts.utils import load_data
from scripts.utils import SimConfig
import tensorflow as tf
import glob

if __name__ == "__main__":
    hdf5_files = glob.glob(f'data/alm_cache/*.hdf5', recursive=True)
    for hdf5_file in hdf5_files:
        ldata = load_data(hdf5_file, ["alm", "almng"])

        alms = ldata["alm"]
        almngs = ldata["almng"]
        fnls = uniform(-1000, 1000, 1000)[:, None, None]

        alm_total = alms + fnls * almngs

        # TODO, convert to pol last just to match conv norms
        # add data and target names to clear up confusion
        # double check the shape of the data
        dataset = tf.data.Dataset.from_tensor_slices((alm_total, fnls))
        tf.data.Dataset.save(dataset, hdf5_file.replace('.hdf5', '.tfds'))



