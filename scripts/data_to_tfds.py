import os
import glob
import sys

import tensorflow as tf
import h5py
import numpy as np
from numpy.random import uniform

from utils import load_data

def convert_hdf5_to_tfdata():
    dirs = ("data/unlensed", "data/lensed")
    for d in dirs:
        hdf5_files = glob.glob(f'{d}/*.hdf5', recursive=True)
        for hdf5_file in hdf5_files:
            # Replace .hdf5 with .tfds in the file path
            tfds_dir = hdf5_file.replace('.hdf5', '.tfds')

            # Check if the .tfds directory does not exist
            if not os.path.isdir(tfds_dir):
                # Convert the .hdf5 file to .tfds
                print('Converting', hdf5_file, 'to', tfds_dir)
                with h5py.File(hdf5_file, 'r') as f:
                    images = f['patches'][:]
                    labels = f['fnls'][:]

                nsims, npatchs, npol, ndup, nside, _ = images.shape
                images = np.reshape(images, (-1, nside, nside, 1))
                labels = np.repeat(labels, ndup)

                # Convert to tf.data.Dataset
                dataset = tf.data.Dataset.from_tensor_slices((images, labels))
                tf.data.Dataset.save(dataset, tfds_dir)


def convert_alm_to_tfds():
    hdf5_files = glob.glob(f'data/alm_cache/*.hdf5', recursive=True)
    for hdf5_file in hdf5_files:
        print('processing', hdf5_file)
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
        print('finsihed with', hdf5_file)


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == 'alm':
        convert_alm_to_tfds()
    elif arg == 'hdf5':
        convert_hdf5_to_tfdata()
    else:
        print('Invalid argument')
        sys.exit(1)