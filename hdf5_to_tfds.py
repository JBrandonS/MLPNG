import os
import glob

import tensorflow as tf
import h5py
import numpy as np

# Helper script to convert to tf.data.Dataset

def convert_hdf5_to_tfdata(hdf5_filepath, save_filepath):
    with h5py.File(hdf5_filepath, 'r') as f:
        # Assuming that you have two datasets 'images' and 'labels' in the HDF5 file
        images = f['patches'][:]
        labels = f['fnls'][:]

    nsims, npatchs, npol, ndup, nside, _ = images.shape
    images = np.reshape(images, (-1, nside, nside, 1))
    labels = np.repeat(labels, ndup)
    # labels = np.reshape(labels, (-1, 1))

    print('images.shape', images.shape)
    print('labels.shape', labels.shape)

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    tf.data.Dataset.save(dataset, save_filepath)

# Usage:
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
            convert_hdf5_to_tfdata(hdf5_file, tfds_dir)