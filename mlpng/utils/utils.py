import os
import logging
import sys
import h5py
import numpy as np
import healpy as hp

logger = logging.getLogger(__name__)


def setup_logging(
    name=__name__,
    level=logging.INFO,
    scripts_level=None,
    base_level=logging.WARNING,
    handlers=[logging.StreamHandler(sys.stdout)],
):
    """
    Set up logging configuration for the application. scripts_level and base_level will default to level if None.

    Args:
        name (str, optional): The name of the logger. Defaults to __name__.
        level (int, optional): The logging level for the logger. Defaults to logging.DEBUG.
        scripts_level (int, optional): The logging level for any 'scripts' loggers. Defaults to None.
        base_level (int, optional): The logging level for most loggers, includes 3rd party loggers.
        handlers (list, optional): The list of logging handlers. Defaults to [logging.StreamHandler(sys.stdout)].

    Returns:
        logger (logging.Logger): The configured logger object.
    """
    if scripts_level is None:
        scripts_level = level

    if base_level is None:
        base_level = level

    logging.basicConfig(
        level=base_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=handlers,
    )

    logging.getLogger("mlpng").setLevel(scripts_level)
    log = logging.getLogger(name)
    log.setLevel(level)
    return log


def save_data(file_path, data_dict, mode="a"):
    """
    Save data to an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        data_dict (dict): A dictionary containing the data to be saved.
        mode (str, optional): The file mode to use when opening the HDF5 file. Defaults to "x"

    Raises:
        None

    Returns:
        None
    """
    logger.debug("Saving data to %s", file_path)
    with h5py.File(file_path, mode) as hf:
        for key, value in data_dict.items():
            logger.debug("%s: Processing key", key)
            if isinstance(value, dict):
                logger.debug("%s: Processing dict", key)
                grp = hf.create_group(key)
                for k, v in value.items():
                    logger.debug("%s: Processing subkey %s", key, k)
                    grp.create_dataset(k, data=np.array(v))

                continue

            if isinstance(value, np.ndarray):
                logger.debug("%s: Processing ndarray", key)
                if key in hf:
                    logger.debug("%s: Appending to existing", key)
                    # Resize the dataset to accommodate the new data
                    hf[key].resize(  # type: ignore
                        (hf[key].shape[0] + value.shape[0],) + value.shape[1:]  # type: ignore
                    )  # type: ignore
                    # Append the new data
                    hf[key][-value.shape[0] :] = value  # type: ignore
                else:
                    # Create a new dataset for this key
                    logger.debug("%s: Creating new dataset", key)
                    hf.create_dataset(
                        key, data=value, maxshape=(None,) + value.shape[1:]
                    )
            else:
                # For other data types, create a dataset
                logger.debug("Creating dataset for key %s of type %s", key, type(value))
                hf.create_dataset(key, data=value)


def load_data(data_file, keys):
    """
    Load data from an HDF5 file.

    Args:
        data_file (str): The path to the HDF5 file.
        keys (str or list): The key(s) of the data to load.

    Returns:
        dict: A dictionary containing the loaded data, where the keys are the provided key(s) and the values are the corresponding data arrays.

    Raises:
        ValueError: If any of the provided keys are not found in the HDF5 file.
    """
    logger.info("Loading data %s from %s", keys, data_file)

    if isinstance(keys, str):
        keys = [keys]

    data = {}
    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        for key in keys:
            logger.debug("%s: Loading", key)
            kv = hdf.get(key, None)
            if kv is None:
                raise ValueError(f"Key {key} not found in {data_file}")
            logger.debug(
                "%s: Loaded, %s", key, kv.shape if isinstance(kv, np.ndarray) else kv
            )
            data[key] = kv[()]  # type: ignore

    logger.info("Finished loading data from %s", data_file)
    return data


def get_fisher(file, name="fisher"):
    """
    Load the fisher matrix from an HDF5 file.

    Parameters:
    - file (str): The path to the HDF5 file.

    Returns:
    - fisher (numpy.ndarray or None): The loaded fisher matrix, or None if it couldn't be loaded.
    """
    try:
        with h5py.File(file, "r", swmr=True, locking=False) as hdf:
            fisher = float(hdf.get(name, [None])[()])  # type: ignore
            logger.info(
                "Loaded fisher: %s, with error: %s", fisher, np.sqrt(1 / fisher)
            )
    except Exception as e:
        logger.error("Could not load fisher matrix: %s", e)
        raise e
    return fisher


def remove_mono_dipole(alm, inplace=False):
    """
    Remove the monopole and dipole terms from the alms.

    Parameters:
        alm (array-like): The input alms.
        inplace (bool, optional): If True, the input alms will be modified in-place.
                                If False, a copy of the input alms will be made before modification.
                                Default is False.

    Returns:
        array-like: The alms with the monopole and dipole terms removed.
    """
    data = alm if inplace else alm.copy()
    lmax = hp.Alm.getlmax(len(data))

    # Note: that we do not need -m's due to symmetry
    if len(np.shape(data)) == 1:
        data[hp.Alm.getidx(lmax, 0, 0)] = 0.0
        data[hp.Alm.getidx(lmax, 1, 0)] = 0.0
        data[hp.Alm.getidx(lmax, 1, 1)] = 0.0
    else:
        data[..., hp.Alm.getidx(lmax, 0, 0)] = 0.0
        data[..., hp.Alm.getidx(lmax, 1, 0)] = 0.0
        data[..., hp.Alm.getidx(lmax, 1, 1)] = 0.0
    return data


def print_errors(truth, preds, fisher, n_sigma=5):
    truth = truth.flatten()
    preds = preds.flatten()

    diff = preds - truth
    std_dev = np.sqrt(1 / fisher)
    sem = std_dev / np.sqrt(len(diff))
    logger.info("Standard deviation: %s, SEM: %s", std_dev, sem)
    for i in range(n_sigma):
        within = np.sum(np.abs(diff) < ((i + 1) * std_dev))
        m_error = np.sum(np.abs(diff) < ((i + 1) * (std_dev - sem)))
        p_error = np.sum(np.abs(diff) < ((i + 1) * (std_dev + sem)))
        logger.info(
            "%s%% (%s, %s) are within %s standard deviations",
            within / len(diff) * 100,
            m_error / len(diff) * 100,
            p_error / len(diff) * 100,
            i + 1,
        )


def trim_alms(alm, lmax, dtype=None):
    """
    Trim the alm array to the specified lmax.

    Parameters:
        alm (np.array): The input alm array, can but of any shape as long as the data is the last dimension and pols are the second to last.
        lmax (int): The maximum l value to keep.
        dtype (np.dtype, optional): The data type to use for the trimmed alms. If None, the same data type as the input alms will be used.

    Returns:
        array-like: The trimmed alm array.
    """
    alm = np.atleast_2d(alm)
    old_lmax = hp.Alm.getlmax(alm.shape[-1])
    dtype = alm.dtype if dtype is None else dtype

    if old_lmax < lmax:
        raise ValueError(
            f"Cannot trim alms to lmax {lmax} when original lmax is {old_lmax}"
        )

    # make the new alms
    new_shape = list(alm.shape)
    new_shape[-1] = hp.Alm.getsize(lmax)  # replace the last dim
    temp = np.zeros(new_shape, dtype)

    # this is probably slow, but dont think it worth fixing yet
    for l in range(lmax + 1):  # noqa: E741
        for m in range(l + 1):
            new = hp.Alm.getidx(lmax, l, m)
            old = hp.Alm.getidx(old_lmax, l, m)
            temp[..., new] = alm[..., old]

    return temp


def try_init_wandb(
    project="mlpng",
    notes=None,
    tags=[],
    config=None,
    dir="data",
    append_to=None,
    patch_tb=False,
    patch_logdir=None,
    **kwargs,
):
    """
    Initialize and configure the Weights & Biases (wandb) library for logging experiments.

    Parameters:
    - project (str): The name of the project to which the run belongs. Default is "mlpng".
    - notes (str): A description or notes for the run. Default is None.
    - tags (list): A list of tags to associate with the run. Default is an empty list.
    - config (dict): A dictionary of configuration parameters for the run. Default is None.
    - dir (str): The directory where the run files will be saved. Default is "data".
    - append_to (list): A list of callbacks to which the WandbMetricsLogger callback will be appended. Default is None.
    - patch_tb (bool): Whether to patch TensorBoard logging. Default is False.
    - patch_logdir (str): The root log directory for patching TensorBoard logging. Required if patch_tb is True.
    - **kwargs: Additional keyword arguments to pass to wandb.init().

    Returns:
    - WandbMetricsLogger: The WandbMetricsLogger callback object.

    Raises:
    - ValueError: If patch_tb is True but patch_logdir is not set.
    """
    try:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger

        logger.debug("wandb version: %s", wandb.__version__)
    except ImportError:
        logger.error(
            "Wandb is not installed! "
            "Please install with `pip install wandb`. "
            "See: https://docs.wandb.ai/quickstart"
        )
        return

    if patch_tb:
        if patch_logdir is None:
            raise ValueError("If patch_tb is True, patch_logdir must be set.")
        wandb.tensorboard.patch(root_logdir=patch_logdir)

    wandb.init(
        project=project, notes=notes, tags=tags, config=config, dir=dir, **kwargs
    )

    # Add the wandb logger to the callbacks, so it is used
    wandb_logger = WandbMetricsLogger()
    if append_to:
        append_to.append(wandb_logger)
    return wandb_logger
