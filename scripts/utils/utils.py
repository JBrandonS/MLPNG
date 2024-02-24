import json
import logging
import os
import sys

import h5py
import numpy as np


def setup_logging(
    name=__name__,
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    set_base=False
):
    if set_base:
        logging.basicConfig(
            # level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=handlers,
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


_logger = setup_logging(__name__)


def safe_makedir(dir):
    "Create a directory if it does not exist. Handles a race condition"
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            _logger.debug("Created directory %s", dir)
        except FileExistsError:
            pass


def save_data(file_path, data_dict):
    _logger.debug("Saving data to %s", file_path)

    with h5py.File(file_path, "a") as hf:
        for key, value in data_dict.items():
            if isinstance(value, dict):
                grp = hf.create_group(key)
                for k, v in value.items():
                    grp[k] = json.dumps(v)

                continue

            if key in hf:
                # Resize the dataset to accommodate the new data
                hf[key].resize((hf[key].shape[0] + value.shape[0],) + value.shape[1:])  # type: ignore
                # Append the new data
                hf[key][-value.shape[0] :] = value  # type: ignore
            else:
                # Create a new dataset for this key
                hf.create_dataset(key, data=value, maxshape=(None,) + value.shape[1:])
    _logger.debug("done")


def load_data(data_file, keys, start_index=None, end_index=None):
    if isinstance(keys, str):
        keys = [keys]

    data = {}
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        for key in keys:
            _logger.debug("Loading key %s from %s", key, data_file)
            kv = hdf.get(key, None)
            if kv is None:
                raise ValueError(f"Key {key} not found in {data_file}")

            if start_index is not None and end_index is not None:
                data[key] = np.array(kv[start_index:end_index])  # type: ignore
            else:
                data[key] = np.array(kv[()])  # type: ignore
    _logger.debug("Finished loading data from %s", data_file)
    return data


def load_single_data(data_file, key, index):
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        kv = hdf.get(key, None)
        if kv is None:
            raise ValueError(f"Key {key} not found in {data_file}")
        else:
            return np.array(kv[index])  # type: ignore


def get_fisher(data_file):
    """Get the fisher matrix from the data file"""
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        kv = hdf.get("fisher", None)
        if kv is None:
            raise ValueError(f"Key fisher not found in {data_file}")
        return np.array(kv[()])  # type: ignore
