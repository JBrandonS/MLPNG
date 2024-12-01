import logging
import h5py
import numpy as np
import healpy as hp

from keras.utils import Sequence

logger = logging.getLogger(__name__)


class HDF5Dataset(Sequence):
    def __init__(
        self,
        file_path,
        x_name="alm",
        y_name="fnl",
        start_idx=0,
        end_idx=None,
        **kwargs,
    ):
        super().__init__()

        # setup some attributes
        self.file_path = file_path
        self.file = h5py.File(file_path, mode="r", swmr=True, locking=False)
        self.x_data = self.file[x_name]
        self.y_data = self.file[y_name]
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else self.x_data.shape[0]  # type: ignore

    def __len__(self):
        # return the number of batches
        return self.end_idx - self.start_idx
        # return self.n_batches

    def __getitem__(self, idx):
        # get the indexes for the batch
        idx += self.start_idx  # self.indexes[idx]

        x_data = []
        for x in self.x_data[idx]:  # type: ignore
            # TODO: fix up
            data = hp.alm2map(x, nside=128, pol=False)
            ring = hp.reorder(data, r2n=True)  # Convert from RING to NEST ordering
            x_data.append(ring)

        data = np.array(x_data).T
        labels = np.array(self.y_data[idx])  # type: ignore
        return data, labels

    def split(self, train_size=0.8, val_size=0.1, test_size=0.1, verbose=False):
        # split the data into train, val, test
        n = self.end_idx - self.start_idx
        train_end = self.start_idx + int(n * train_size)
        val_end = train_end + int(n * val_size)
        test_end = np.min((val_end + int(n * test_size), self.end_idx))

        if verbose:
            logger.info(
                "Splitting data into train: %s, val: %s, test: %s",
                train_end - self.start_idx,
                val_end - train_end,
                test_end - val_end,
            )

        train = HDF5Dataset(self.file_path, start_idx=self.start_idx, end_idx=train_end)
        val = HDF5Dataset(self.file_path, start_idx=train_end, end_idx=val_end)
        test = HDF5Dataset(self.file_path, start_idx=val_end, end_idx=test_end)
        return train, val, test
