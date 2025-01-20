import logging
import h5py
import numpy as np

from keras.utils import Sequence

logger = logging.getLogger(__name__)


class HDF5Dataset(Sequence):

    def __init__(
        self,
        file_path,
        x_name="map",
        y_name="fnl",
        start_idx=0,
        end_idx=None,
        verbose=True,
    ):
        super().__init__()

        # setup some attributes
        self.file_path = file_path
        self.file = h5py.File(file_path, mode="r", swmr=True, locking=False)

        # we dont actually limit the data here because that will load into memory
        self.x_data = self.file[x_name]
        self.y_data = self.file[y_name]
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else self.x_data.shape[0]  # type: ignore
        if verbose:
            logger.debug("Loaded HDF5 dataset: '%s'", file_path)
            logger.debug("Found keys: %s", self.file.keys())
            logger.debug("Data shape: (%s, %s)", self.x_data.shape, self.y_data.shape)  # type: ignore

    def __len__(self):
        # we batch later so this is the len of the full DS
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # get the indexes for the batch
        idx += self.start_idx
        data = np.array(self.x_data[idx])  # type: ignore
        labels = np.array(self.y_data[idx]).flatten()  # type: ignore
        return data, labels

    def split(self, train_size=0.8, val_size=0.1, test_size=0.1):
        # split the data into train, val, test
        n = self.end_idx - self.start_idx
        train_end = self.start_idx + int(n * train_size)
        val_end = train_end + int(n * val_size)
        test_end = np.min((val_end + int(n * test_size), self.end_idx))

        logger.debug(
            "Splitting data into train: %s, val: %s, test: %s",
            train_end - self.start_idx,
            val_end - train_end,
            test_end - val_end,
        )

        train = HDF5Dataset(self.file_path, start_idx=self.start_idx, end_idx=train_end)
        val = HDF5Dataset(self.file_path, start_idx=train_end, end_idx=val_end)
        test = HDF5Dataset(self.file_path, start_idx=val_end, end_idx=test_end)
        return train, val, test
