import sys
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


def compute_iso_icov(core, N, b):
    # Inverse covariance matrix diagonal in ell. Unlike "icov" this should be: 1 / (S_ell + (b^{-1} N b^{-1})_ell)
    S_ell = core.cosmo._camb_data.get_cmb_power_spectra(
        core.cosmo.camb_params,
        core.lmax,
        ["total"],
        "muK",
        True,
    )["total"]
    S_ell = np.transpose(S_ell, (1, 0))[: core.npol]
    b_inv = 1 / b
    return 1 / (S_ell + b_inv * N * b_inv)


def run_ksw_step(core, theta_batch):
    def step_loader(idx):
        """
        for stepping the KSW estimator, we just generate new unique sims
        """
        logger.debug("Sending alm step %s", idx)
        return core.data.compute_alm_sim(core.lensing)

    logger.info("Running KSW step")

    # lets get our iso fisher so we can find the base line
    icov_ell = compute_iso_icov(core, core.noise_ell[: core.npol], core.beam_ell)
    fisher_iso = core.ksw.compute_fisher_isotropic(icov_ell)
    # N_fact = core.cosmo.red_bispectra[0].nfact

    core.ksw.step_batch(
        step_loader, np.arange(100), comm=mpi_comm, theta_batch=theta_batch
    )

    # compute the new fisher and the distance
    fisher = core.ksw.compute_fisher() / core.npol  # note bug fix here
    distance = np.abs(fisher - fisher_iso)
    logger.debug("Distance: %s", distance)


def main():
    core = Core()

    # the KSW code requires the total_sims to be >= mpi_size
    # best usage would have total_sims % mpi_size == 0, but not required
    assert (
        core.total_sims >= mpi_size
    ), "total_sims < mpi_size, lower ntasks or increase sims"

    # early loading to fail fast if the file does not exist
    alm_file = h5py.File(core.alm_file, "r", swmr=True, locking=False)

    # The default theta_batch size is 25, which is really small, we want to increase it
    # going too high can cause memory issues, so we will cap it at 256
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    theta_batch = min(256, theta_batch)
    logger.debug("Using theta_batch %s", theta_batch)

    # check for existing ksw state, if it exists, load it
    # otherwise, run the MC, can take a few hours
    use_mc_file = True  # just a quick disable
    mc_path = os.path.join(core.base_dir, "kswmc")
    os.makedirs(mc_path, exist_ok=True)
    mc_file = os.path.join(mc_path, f"{core.base_name}.hdf5")

    # remove the file to force its recreation
    if core.force_ksw and use_mc_file and os.path.exists(mc_file):
        logger.info("Removing existing KSW state")
        os.remove(mc_file)

    if use_mc_file and os.path.exists(mc_file):
        logger.info("Loading KSW state from %s", mc_file)
        core.ksw.start_from_read_state(mc_file, mpi_comm)
    else:
        run_ksw_step(core, theta_batch)

        # save the mc state if we are using the mc file
        if use_mc_file and mpi_root:
            logger.info("Saving KSW state to %s", mc_file)
            core.ksw.write_state(mc_file, mpi_comm)

    # NOTE: There seems to be an issue with the fisher computation with TE pols being a factor 2 higher than expected
    # I do not know why but this seems to fix the issue
    # TODO: This is a hack, we need to figure out why the TE pols are off by a factor of 2
    fisher = float(core.ksw.compute_fisher()) / core.npol
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory
    alms = alm_file["alm"]
    fnls = alm_file["fnl"]

    num_est = min(core.num_estimates, core.total_sims)
    idxs = np.arange(num_est)
    logger.info(
        "Computing %s estimates in %.2f batches",
        num_est,
        num_est / mpi_size,
    )

    def estimator_loader(idx):
        """Loads in a single alm given an idx."""
        logger.debug("Sending %s with fnl %s", idx, fnls[idx])
        return np.array(alms[idx])

    estimates = core.ksw.compute_estimate_batch(
        estimator_loader,
        idxs,
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        verbose=True,
    )

    if mpi_root:
        logger.info("Saving data")

        fnls = np.array(fnls[idxs]).flatten()

        # alm_file is read only and we need to append to it
        # close it, so we can open in append mode
        alm_file.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = [fisher]
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(core.alm_file, sdata, mode="a")

        # make and save some plots
        plot_dir = os.path.join(core.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)

        pred_file = os.path.join(plot_dir, f"{core.sjob}_{core.base_name}_preds.png")
        plot_predictions(fnls, estimates, fisher=fisher, save_file=pred_file)

        hist_file = os.path.join(plot_dir, f"{core.sjob}_{core.base_name}_hist.png")
        plot_histogram(fnls, estimates, save_file=hist_file)

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
