import logging
import sys
import inspect
import re
import h5py
import numpy as np
import healpy as hp

logger = logging.getLogger(__name__)


def setup_logging(
    name=__name__,
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    set_base=True,
    base_level=None,
    use_rich=True,
):
    if set_base:
        if use_rich:
            from rich.logging import RichHandler

            handlers = [
                RichHandler(
                    show_time=False,
                    show_level=False,
                    show_path=False,
                    rich_tracebacks=True,
                    locals_max_length=3,
                    locals_max_string=None,
                )
            ]

        logging.basicConfig(
            level=base_level or level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=handlers,
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def save_data(file_path, data_dict, mode="x"):
    logger.info("Saving data to %s", file_path)
    with h5py.File(file_path, mode) as hf:
        for key, value in data_dict.items():
            logger.debug("Processing key %s", key)
            if isinstance(value, dict):
                logger.debug("Processing dict for key %s", key)
                grp = hf.create_group(key)
                for k, v in value.items():
                    # grp[k] = json.dumps(v)
                    logger.debug("Processing subkey %s", k)
                    grp.create_dataset(k, data=np.array(v))

                continue

            if isinstance(value, np.ndarray):
                logger.debug("Processing ndarray for key %s", key)
                if key in hf:
                    logger.debug("Appending to existing dataset for key %s", key)
                    # Resize the dataset to accommodate the new data
                    hf[key].resize((hf[key].shape[0] + value.shape[0],) + value.shape[1:])  # type: ignore
                    # Append the new data
                    hf[key][-value.shape[0] :] = value  # type: ignore
                else:
                    # Create a new dataset for this key
                    logger.debug("Creating new dataset for key %s", key)
                    hf.create_dataset(
                        key, data=value, maxshape=(None,) + value.shape[1:]
                    )
            else:
                # For other data types, create a dataset
                logger.debug("Creating dataset for key %s of unknown type", key)
                hf.create_dataset(key, data=value)
    logger.debug("Finished saving data to %s", file_path)


def load_data(data_file, keys):
    logger.info("Loading data %s from %s", keys, data_file)

    if isinstance(keys, str):
        keys = [keys]

    data = {}
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        for key in keys:
            logger.debug("Loading key %s", key)
            kv = hdf.get(key, None)
            if kv is None:
                raise ValueError(f"Key {key} not found in {data_file}")
            logger.debug(
                "Loaded key %s, %s", key, kv.shape if isinstance(kv, np.ndarray) else kv
            )
            data[key] = kv[()]  # type: ignore

    logger.info("Finished loading data from %s", data_file)
    return data


def log_source(func):
    source = inspect.getsource(func)
    source = re.sub(r"#.*", "", source)
    source = re.sub(r"\n\s*\n", "\n", source)
    logger.info(f"Model source:\n{source}")


def remove_mono_dipole(alm):
    """
    Remove the monopole and dipole terms from the alms.
    Note that we do not need -m's due to symmetry
    """
    lmax = hp.Alm.getlmax(len(alm))
    alm[..., hp.Alm.getidx(lmax, 0, 0)] = 0.0
    alm[..., hp.Alm.getidx(lmax, 1, 0)] = 0.0
    alm[..., hp.Alm.getidx(lmax, 1, 1)] = 0.0
    return alm
