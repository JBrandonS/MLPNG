import logging
import sys
import warnings

import h5py
import healpy as hp
import numpy as np


def setup_logging(
    name=__name__,
    level=logging.INFO,
    scripts_level=None,
    base_level=None,
    handlers=None,
):
    """
    Set up logging configuration for the application. scripts_level and base_level will default to level if None.

    This function should be called once at the entry point of your application (e.g., in `if __name__ == "__main__"`
    blocks or in notebook first cells). Library modules should use `logging.getLogger(__name__)` instead.

    Args:
        name (str, optional): The name of the logger. Defaults to __name__.
        level (int, optional): The logging level for the logger. Defaults to logging.INFO.
        scripts_level (int, optional): The logging level for any 'mlpng' loggers. Defaults to None (uses level).
        base_level (int, optional): The logging level for most loggers, includes 3rd party loggers.
        handlers (list, optional): The list of logging handlers. Defaults to None (creates StreamHandler to stdout).

    Returns:
        logger (logging.Logger): The configured logger object.
    """
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    if scripts_level is None:
        scripts_level = level

    if base_level is None:
        base_level = level

    # Configure root logger, clearing any existing handlers first to prevent duplicates
    # This is especially important in Jupyter notebooks which add their own handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if handlers is None:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=base_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logging.getLogger("mlpng").setLevel(scripts_level)
    log = logging.getLogger(name)
    log.setLevel(level)
    return log


def recursive_save(file, path, data, verbose=True):
    """
    Recursively save a dictionary to an HDF5 file.

    Args:
        file (h5py.File): The HDF5 file object.
        path (str): The path in the HDF5 file to save the dictionary.
        dict (dict): The dictionary to save.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    for key, value in data.items():
        npath = f"{path}/{key}"
        if verbose:
            logger.debug("Processing path: %s", npath)

        if isinstance(value, dict):
            if verbose:
                logger.debug("Processing dict: %s", key)

            # check if group exists, if not create it
            if npath in file:
                grp = file[npath]
            else:
                grp = file.create_group(npath)
            recursive_save(grp, npath, value, verbose)

        elif isinstance(value, np.ndarray):
            # logger.debug("Processing ndarray: %s", key)
            if npath in file:
                if verbose:
                    logger.debug("Appending %s/%s data to existing", path, key)
                # Resize the dataset to accommodate the new data
                file[npath].resize(
                    (file[npath].shape[0] + value.shape[0],) + value.shape[1:]
                )
                file[npath][-value.shape[0] :] = value
            else:
                if verbose:
                    logger.debug("Creating new dataset: %s/%s", path, key)
                file.create_dataset(
                    npath, data=value, maxshape=(None,) + value.shape[1:]
                )
        else:
            if verbose:
                logger.debug(
                    "Creating dataset for key %s of type %s at %s",
                    key,
                    type(value),
                    path,
                )
            file.create_dataset(npath, data=value)


def save_data(file_path, data_dict, mode="a", verbose=False):
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
    logger = logging.getLogger(__name__)

    if verbose:
        logger.info("Saving data to '%s'", file_path)
    with h5py.File(file_path, mode) as hf:
        recursive_save(hf, "", data_dict, verbose=verbose)
    if verbose:
        logger.debug("Finished saving data to '%s'", file_path)


def get_ksw_save_data(ksw):
    """
    Extracts the data to be saved from a KSW object.

    Mostly this is just a modified ksw.estimator.write_state

    Parameters:
        ksw (KSW): The KSW object containing the data.

    Returns:
        dict: A dictionary containing the data to be saved.
    """

    mc_idx = np.asarray([ksw.mc_idx], dtype=np.int64)

    if ksw.__mc_gt_sq is None:
        mc_gt_sq = np.asarray([np.nan], dtype=np.float64)
    else:
        mc_gt_sq = np.asarray([ksw.__mc_gt_sq], dtype=np.float64)

    if ksw.__mc_gt is None:
        mc_gt = np.asarray([np.nan], dtype=ksw.cdtype)
    else:
        mc_gt = ksw.__mc_gt

    return {"mc_idx": mc_idx, "mc_gt_sq": mc_gt_sq, "mc_gt": mc_gt}


def get_ksw_from_data(data):
    """
    Extracts the KSW object from the saved data.

    TODO: Needs testing

    Parameters:
        data (dict): The dictionary containing the saved data.

    Returns:
        KSW: The KSW object reconstructed from the saved data.
    """
    from mlpng.estimators.ksw import KSW

    mc_idx = data["mc_idx"][0]
    mc_gt_sq = data["mc_gt_sq"][0]
    mc_gt = data["mc_gt"]

    ksw = KSW(mc_idx=mc_idx, mc_gt_sq=mc_gt_sq, mc_gt=mc_gt)
    return ksw


def get_data(file_path, key, idxs=None):
    """
    Retrieve data from an HDF5 file. With error checking to keep the linter happy.

    Args:
        file_path (str): The path to the HDF5 file.
        key (str): The key to retrieve data from the file.

    Returns:
        np.ndarray: The data retrieved from the file.
    """
    with h5py.File(file_path, "r", swmr=True, locking=False) as hf:
        if key in hf:
            data = hf[key]

            # handle a few possible errors in the keys
            if isinstance(data, h5py.Group):
                raise KeyError(
                    f"Key '{key}' refers to a group, not a dataset. Please provide the full path to a dataset within this group. Group keys: {list(data.keys())}"
                )
            elif not isinstance(data, h5py.Dataset):
                raise TypeError(
                    f"Key '{key}' does not refer to a dataset but instead a {type(data)}. Please provide a valid dataset key."
                )

            # finally get the data, load in the idxs if provided and return as np array
            return np.array(data[:] if idxs is None else data[np.array(idxs)])
        else:
            # key isn't found in dataset, so we strip the leading path and print with possible keys for clear logging
            sub = "/".join(key.split("/")[:-1])
            raise KeyError(
                f"Key '{key}' not found in file '{file_path}'. Possible subgroups for {sub} are {list(hf[sub].keys())}"  # type: ignore
            )


def remove_mono_dipole(alm, inplace=False):
    """
    Remove the monopole and dipole terms from the alms.

    Parameters:
        alm (array-like): The input alms, either 1d array or any number of dimensions with the last dimension being the alm data.
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
    return np.ascontiguousarray(data)


def print_errors(truth, preds, fisher, n_sigma=5):
    """
    Print the errors between the truth and predictions, along with statistics.
    Parameters:
        truth (np.ndarray): The ground truth values.
        preds (np.ndarray): The predicted values.
        fisher (float): The Fisher information value.
        n_sigma (int, optional): The number of standard deviations to consider. Default is 5.
    """
    logger = logging.getLogger(__name__)

    truth = truth.flatten()
    preds = preds.flatten()

    diff = preds - truth
    std_dev = np.sqrt(1 / fisher)
    sem = std_dev / np.sqrt(len(diff))

    logger.info("Standard deviation from fisher: %s, SEM: %s", std_dev, sem)
    logger.info("Mean error: %s, Median error: %s", np.mean(diff), np.median(diff))
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
    logger = logging.getLogger(__name__)

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
        return None

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
