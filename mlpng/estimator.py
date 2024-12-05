import sys
import logging
import os
import healpy as hp

import h5py
import numpy as np
from mpi4py import MPI

from . import Core, get_itotcov_ell
from .generator import generate_alm
from .utils import save_data, setup_logging, print_errors
from .utils.plots import plot_histogram, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

# estimator will run multiple jobs per id which get sent to the same log file,
# so we only want to log the root to keep from spamming the log
if mpi_root:
    logger = setup_logging(name=f"mlpng.estimator_{mpi_rank}", level=logging.DEBUG)
else:
    logger = setup_logging(name=f"mlpng.estimator_{mpi_rank}", level=logging.ERROR)


def icov_func(icov, alm):
    ret = np.zeros_like(alm)
    for pol in range(ret.shape[0]):
        ic = icov[pol, pol] if icov.ndim == 3 else icov[pol]
        ret[pol] = hp.almxfl(alm[pol], ic)
    return ret


def main():
    core = Core()

    # the KSW code requires the total_sims to be >= mpi_size
    # best usage would have total_sims % mpi_size == 0, but not required
    assert (
        core.num_estimates >= mpi_size
    ), "total_sims < mpi_size, lower ntasks or increase sims"

    logger.info(
        "Computing %s estimates in %.2f batches",
        core.num_estimates,
        core.num_estimates / mpi_size,
    )
    if core.num_estimates % mpi_size != 0:
        logger.warning(
            "num_estimates is not divisible by mpi_size, "
            "this will lead to uneven workloads."
        )

    # early loading to fail fast if the file does not exist
    data = h5py.File(core.file, "r", swmr=True, locking=False)

    core.init_estimator()

    # The default theta_batch size is 25, which is really small, we want to increase it
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    logger.debug("Using theta_batch %s", theta_batch)

    # get our icov
    pols = core.pol_idxs(keep_b=False, keep_te=False, pretrimmed=True)
    ic_ell = np.zeros_like(core.c_ell)
    ic_ell[pols, core.lmin :] = 1 / core.c_ell[pols, core.lmin :]
    inoise = np.full(core.n_ell.shape, 1e-16)
    inoise[pols, core.lmin :] = 1 / core.n_ell[pols, core.lmin :]
    icov = get_itotcov_ell(ic_ell, inoise, core.b_ell)
    icov = np.array([icov[i, i] for i in pols])

    if core.isotropic:
        # TODO: We can just read in the fisher_iso value instead of generating again
        logger.debug("Computing the isotropic fisher")
        fisher = core.estimator.compute_fisher_isotropic(icov, comm=mpi_comm)
    else:
        logger.debug("Computing fisher")
        if os.path.exists(core.mc_file):
            logger.info("Loading KSW state from %s", core.mc_file)
            core.estimator.start_from_read_state(core.mc_file, comm=mpi_comm)
        else:
            alm_steps = generate_alm(core, nsims=core.mc_steps)

            def step_loader(idx):
                """for stepping the KSW estimator, we just generate new unique sims"""
                logger.debug("Sending alm step %s", idx)
                return alm_steps[idx]

            logger.info("Initializing KSW with %s steps", core.mc_steps)
            core.estimator.step_batch(
                step_loader,
                range(core.mc_steps),
                mpi_comm,
                theta_batch=theta_batch,
            )

            # save the mc state if we are using the mc file
            if core.slurm.is_main:
                logger.info("Saving KSW state to %s", core.mc_file)
                core.estimator.write_state(core.mc_file, comm=mpi_comm)

        fisher = core.estimator.compute_fisher()
    logger.info("Fisher: %s, standard deviation: %s", fisher, 1 / np.sqrt(fisher))

    # Finally we can get our estimates
    pol_idxs = core.pol_idxs(pretrimmed=True)
    # TODO: double check that KSW doesnt support lensed data
    # alms = data["alm_lensed"] if core.lensing else
    alms = data["alm"]  # not in memory yet

    logger.debug("Computing estimates with alms of shape %s, dtype: %s", alms.shape, alms.dtype)
    logger.debug("Right now the code will only use the first duplicate background")
    estimates, _, _, _ = core.estimator.compute_estimate_batch(
        lambda idx: icov_func(icov, alms[idx, 0, pol_idxs]),
        range(core.num_estimates),
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        lin_term=0 if core.isotropic else None,
    )

    fnls = np.array(data["fnl"][: core.num_estimates])[:, 0].flatten()
    print_errors(fnls, estimates, fisher)

    if mpi_root:
        # save the data, this will append to the alm_file
        # TODO fisher is fisher_iso if core.isotropic, does this matter?
        sdata = {}
        sdata["fisher"] = fisher
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)

        # We need to close the data file before we can write to it as it is opened in read-only
        data.close()

        logger.info("Appending estimator data to %s", core.file)
        save_data(core.file, sdata, mode="a")

        if core.plot:
            plot_predictions(
                fnls, estimates, fisher=fisher, save_file=core.get_plot_file("preds")
            )
            plot_histogram(fnls, estimates, save_file=core.get_plot_file("hist"))

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
