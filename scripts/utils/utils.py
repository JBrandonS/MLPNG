import json
import logging
import os
import sys
import time
import inspect
import re
import h5py
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(
    name=__name__,
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    set_base=True,
):
    if set_base:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=handlers,
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def save_data(file_path, data_dict):
    start_time = time.perf_counter()
    logger.info("Saving data to %s", file_path)
    with h5py.File(file_path, "a") as hf:
        for key, value in data_dict.items():
            if isinstance(value, dict):
                grp = hf.create_group(key)
                for k, v in value.items():
                    # grp[k] = json.dumps(v)
                    grp.create_dataset(k, data=np.array(v))

                continue

            if isinstance(value, np.ndarray):
                if key in hf:
                    # Resize the dataset to accommodate the new data
                    hf[key].resize((hf[key].shape[0] + value.shape[0],) + value.shape[1:])  # type: ignore
                    # Append the new data
                    hf[key][-value.shape[0] :] = value  # type: ignore
                else:
                    # Create a new dataset for this key
                    hf.create_dataset(key, data=value, maxshape=(None,) + value.shape[1:])
            else:
                # For other data types, create a dataset
                hf.create_dataset(key, data=np.array(value))
    logger.info("Finished saving data to %s in %s", file_path, time.perf_counter() - start_time)


def load_data(data_file, keys):
    start_time = time.perf_counter()
    logger.info("Loading data %s from %s", keys, data_file)

    if isinstance(keys, str):
        keys = [keys]

    data = {}
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        for key in keys:
            logger.debug("Loading key %s from %s", key, data_file)
            kv = hdf.get(key, None)
            if kv is None:
                raise ValueError(f"Key {key} not found in {data_file}")
            data[key] = kv # type: ignore

    logger.info("Finished loading data from %s in %s", data_file, time.perf_counter() - start_time)
    return data

def log_source(func):
    source = inspect.getsource(func)
    source = re.sub(r"#.*", "", source)
    source = re.sub(r"\n\s*\n", "\n", source)
    logger.info(f"Model source:\n{source}")