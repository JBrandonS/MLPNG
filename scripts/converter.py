import os
import sys

from tqdm.auto import tqdm
from itertools import product

import tensorflow as tf
import h5py
import numpy as np
from healpy.sphtfunc import Alm

from utils import Config


def convert_hdf5_to_tfdata(hdf5_file):
    if not os.path.exists(hdf5_file):
        print("File not found:", hdf5_file)
        return
    
    # Replace .hdf5 with .tfds in the file path
    tfds_dir = hdf5_file.replace(".hdf5", ".tfds")
    if not os.path.isdir(tfds_dir):
        # Convert the .hdf5 file to .tfds
        print("Converting", hdf5_file, "to", tfds_dir)
        with h5py.File(hdf5_file, "r") as f:
            images = np.array(f["patches"][:])
            labels = np.array(f["fnls"][:])

        nsims, npatchs, npol, ndup, nside, _ = images.shape
        images = np.reshape(images, (-1, nside, nside, 1))
        labels = np.repeat(labels, ndup)

        # Convert to tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        tf.data.Dataset.save(dataset, tfds_dir)
    else:
        print("Found existing", tfds_dir, "skipping", hdf5_file)


def convert_alm_to_tfds(hdf5_file):
    if not os.path.exists(hdf5_file):
        print("File not found:", hdf5_file)
        return

    tfds_dir = hdf5_file.replace(".hdf5", ".tfds")
    if not os.path.isdir(tfds_dir):
        print("processing", hdf5_file)
        ldata = h5py.File(hdf5_file, mode="r", swmr=True, locking=False)

        alms = ldata["alm"]
        fnls = ldata["fnl"]
        (nsims, npol, ndata) = alms.shape

        lmax = Alm.getlmax(ndata)

        # creates the index map for the data conversion from alm(i) -> alm(l, m)
        idx_map = np.fromfunction(
            lambda l, m: Alm.getidx(lmax, l, m),
            (lmax, lmax),
            dtype=np.int64,
        )

        # creates our positional encoding to give the model some sense of the position of the data
        pos_enc = idx_map / ndata / 1000

        # converts data(i) -> data(l, m), also splits the real and imaginary parts and adds the index map
        def generator():
            for i, j in tqdm(
                product(range(nsims), range(npol)),
                desc="processing",
                total=(nsims * npol),
            ):
                real_part = np.real(alms[i, j, ...][idx_map]) + pos_enc
                imag_part = np.imag(alms[i, j, ...][idx_map]) + pos_enc
                yield (real_part, imag_part), fnls[i, j]

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(2, lmax, lmax), dtype=tf.float32),
                    tf.TensorSpec(shape=(1,), dtype=tf.float32),
                )
            ),
        )
        tf.data.Dataset.save(dataset, tfds_dir)
        print("finsihed with", hdf5_file)
    else:
        print("Found existing", tfds_dir, "skipping", hdf5_file)


if __name__ == "__main__":
    # s = Config()

    # convert_hdf5_to_tfdata(s.patch_file)
    convert_alm_to_tfds(sys.argv[1])
