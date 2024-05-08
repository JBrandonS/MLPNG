import logging
import os
import math

import h5py
import numpy as np
from mpi4py import MPI

from . import Core
from .utils import save_data, setup_logging
from .utils.plots import plot_histogram, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

# estimator will run multiple jobs per id which get sent to the same log file,
# so we only want to log the root to keep from spamming the log
logger = setup_logging(
    name=f"{__name__}_{mpi_rank}", level=logging.DEBUG if mpi_root else logging.ERROR
)


def _estimator_loader(idx):
    """Loads in a single alm given an idx. Used inside the KSW code."""
    idx, pol = np.unravel_index(int(idx), (core.total_sims, core.npol))
    logger.debug("Sending alm (%s, %s) with fnl %s", idx, pol, fnls[idx, pol])
    return np.array(alms[idx, pol])


def _step_loader(idx):
    """
    for stepping the KSW estimator, we just generate new unique sims
    """
    logger.debug("Sending alm step %s", idx)
    return core.data.compute_alm_sim(core.lensing)


if __name__ == "__main__":
    core = Core()

    # the KSW code requires the total_sims to be >= mpi_size
    assert (
        core.total_sims >= mpi_size
    ), "total_sims < mpi_size, lower ntasks or increase sims"

    # early loading to fail fast if the file does not exist
    alm_file = h5py.File(core.alm_file, "r", swmr=True, locking=False)

    # The default theta_batch size is 25, which is really small
    # Calculate the power of two that lmax is greater than 512 and use this to limit size
    divisor = 2 ** max(0, math.ceil(math.log2(core.lmax / 512)))
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // divisor
    logger.debug("Using theta_batch %s", theta_batch)

    # check for existing ksw state, if it exists, load it
    # otherwise, run the MC, can take a few hours
    use_mc_file = True  # just a quick disable
    mc_path = os.path.join(core.alm_dir, "kswmc")
    os.makedirs(mc_path, exist_ok=True)
    mc_file = os.path.join(mc_path, f"{core.base_name}.hdf5")
    if use_mc_file and os.path.exists(mc_file):
        logger.info("Loading KSW state from %s", mc_file)
        core.ksw.start_from_read_state(mc_file, mpi_comm)
    else:
        logger.info("Running KSW step")

        # we only need to step the MC about 100 time to get a good convergence
        # there is an issue with the KSW code when the total number of steps < mpi_size
        # so we just set it to mpi_size if mpi_size > 100
        idxs = range(max(100, mpi_size))

        # step the MC, actually does the work
        core.ksw.step_batch(_step_loader, idxs, comm=mpi_comm, theta_batch=theta_batch)

        # save the mc state if we are using the mc file
        if use_mc_file and mpi_root:
            logger.info("Saving KSW state to %s", mc_file)
            core.ksw.write_state(mc_file, mpi_comm)

    fisher = float(core.ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory
    alms = alm_file["alm"]
    fnls = alm_file["fnl"]

    # only do at most 1k estimates right now, just for time
    num_est = min(1000, core.total_sims)
    logger.info(
        "Computing %s estimates in %.2f batches",
        num_est,
        num_est / mpi_size,
    )
    idxs = np.arange(num_est)
    estimates = core.ksw.compute_estimate_batch(
        _estimator_loader,
        idxs,
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        verbose=True,
    )

    if mpi_root:
        logger.info("Saving data")

        # first, need to close the existing file or we get an error
        fnls = np.array(fnls[idxs]).flatten()

        # alm_file is read only and we need to append to it
        # close it, so we can open in append mode
        alm_file.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = [fisher]
        sdata["estimate_1k"] = True
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(core.alm_file, sdata, mode="a")

        # make and save some plots
        plot_dir = os.path.join(core.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)
        pred_file = os.path.join(plot_dir, f"{core.sjob}_{core.base_name}_preds.png")
        hist_file = os.path.join(plot_dir, f"{core.sjob}_{core.base_name}_hist.png")
        plot_predictions(fnls, estimates, fisher=fisher, save_file=pred_file)
        plot_histogram(fnls, estimates, save_file=hist_file)

    logger.info("Finished %s!", mpi_rank)
