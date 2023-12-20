import numpy as np
import tensorflow as tf


class TFDSDataLoader():
    def __init__(self, file_name, shuffle=True, seed=None, normalize=False):
        """
        Initializes the DataLoader object.

        Args:
            file (str): The path to the file to load data from.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            seed (int, optional): The seed for the random number generator used for shuffling.
                If None, a random seed is generated. Defaults to None.
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
        """
        self.file_name = file_name
        self.shuffle = shuffle
        self.normalize = normalize

        if seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            self.seed = seed

    def _normalize(self, image, label):
        min_val = tf.reduce_min(image)
        max_val = tf.reduce_max(image)
        image = (image - min_val) / (max_val - min_val)
        return image, label

    def _setup_tfds(self, ds, start, step, batch_size=1):
        ret = ds.skip(start).take(step)
        if self.normalize:
            ds = ds.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ret = ret.cache()
        if self.shuffle:
            # use a buffer size of 1000 prevents true shuffling but doesnt load everything into memory
            ret = ret.shuffle(buffer_size=1000, seed=self.seed)
        ret = ret.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        return ret.prefetch(tf.data.AUTOTUNE)

    def get_split_tfdataset(
        self, train_frac=0.8, test_frac=0.1, val_frac=0.1, batch_size=1
    ):
        ds = tf.data.Dataset.load(self.file_name)
        length = ds.cardinality().numpy()

        ntrain = int(train_frac * length)
        ntest = int(test_frac * length)
        nval = int(val_frac * length)

        ds_train = self._setup_tfds(ds, 0, ntrain, batch_size)
        ds_test = self._setup_tfds(ds, ntrain, ntest, batch_size)
        ds_val = self._setup_tfds(ds, ntrain + ntest, nval, batch_size)
        return ds_train, ds_test, ds_val
