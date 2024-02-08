import sys
from numpy.random import uniform

from scripts.utils import load_data
from scripts.utils import SimConfig
import tensorflow as tf
import glob
import numpy as np

if __name__ == "__main__":
    hdf5_files = glob.glob(f'data/alm_cache/*.hdf5', recursive=True)
    for hdf5_file in hdf5_files:
        ldata = load_data(hdf5_file, ["alm", "almng"])

        alms = ldata["alm"]
        almngs = ldata["almng"]
        fnls = uniform(-1000, 1000, (alms.shape[0], alms.shape[1], 1))

        alm_total = alms + fnls * almngs

        # currently in [nsim, npol, data], I will swap so that pol acts as the last dimension
        alm_total = np.swapaxes(alm_total, 1, 2) # [nsim, data, npol]
        print('alm_shape', alm_total.shape)

        dataset = tf.data.Dataset.from_tensor_slices((alm_total, fnls))
        tf.data.Dataset.save(dataset, hdf5_file.replace('.hdf5', '.tfds'))



