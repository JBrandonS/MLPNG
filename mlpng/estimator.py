import sys
import logging
import os
import healpy as hp

import h5py
import numpy as np
from mpi4py import MPI

from . import Core, get_itotcov_ell
from .generator import generate_alm
from .utils import save_data, setup_logging
from .utils.plots import plot_histogram, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

# estimator will run multiple jobs per id which get sent to the same log file,
# so we only want to log the root to keep from spamming the log
if mpi_root:
    logger = setup_logging(name=f"estimator_{mpi_rank}", level=logging.DEBUG)
else:
    logger = setup_logging(name=f"estimator_{mpi_rank}", level=logging.ERROR)


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
        core.total_sims >= mpi_size
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

    # since we are only doing full sky right now, we do not need to use MC methods
    # TODO, check mc methods
    if core.force_ksw:
        if os.path.exists(core.mc_file):
            logger.info("Loading KSW state from %s", core.mc_file)
            core.estimator.start_from_read_state(core.mc_file, comm=mpi_comm)
        else:
            alm_steps = generate_alm(core, nsims=core.mc_steps)
            logger.debug("Done")

            def step_loader(idx):
                """for stepping the KSW estimator, we just generate new unique sims"""
                logger.debug("Sending alm step %s", idx)
                return alm_steps[idx, : core.npols]

            logger.info("Running KSW step, num steps: %s", 100)
            core.estimator.step_batch(
                step_loader,
                range(core.mc_steps),
                mpi_comm,
                theta_batch=theta_batch,
            )

            # save the mc state if we are using the mc file
            logger.info("Saving KSW state to %s", core.mc_file)
            core.estimator.write_state(core.mc_file, comm=mpi_comm)

        fisher = core.estimator.compute_fisher()
    else:
        ic_ell = np.zeros_like(core.c_ell)
        ic_ell[..., core.lmin :] = 1 / core.c_ell[..., core.lmin :]

        inoise = np.full(core.n_ell.shape, 1e-16)
        inoise[..., core.lmin :] = 1 / core.n_ell[..., core.lmin :]

        icov = get_itotcov_ell(ic_ell, inoise, core.b_ell)

        logger.debug("Computing Fisher")
        fisher = core.estimator.compute_fisher_isotropic(icov, comm=mpi_comm)
    logger.info("Fisher: %s, standard deviation: %s", fisher, 1 / np.sqrt(fisher))

    # Finally we can get our estimates
    pol_idxs = core.pol_idxs(pretrimmed=True)
    alms = data["alm_lensed"] if core.lensing else data["alm"]  # not in memory yet

    # look into changing this but its working
    cov = core.c_ell + core.n_ell / core.b_ell**2
    icov = np.zeros_like(core.s_ell)
    icov[..., core.lmin :] = 1 / cov[..., core.lmin :]
    icov[..., : core.lmin] = 0
    icov *= core.b_ell**2  # maybe could do this is cov to save some time

    estimates, _, _, _ = core.estimator.compute_estimate_batch(
        lambda idx: icov_func(icov, alms[idx, pol_idxs]),
        range(core.num_estimates),
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        lin_term=0 if not core.force_ksw else None,
    )

    if mpi_root:
        logger.info("Saving data")

        fnls = np.array(data["fnl"][:])
        plot_predictions(
            fnls, estimates, fisher=fisher, save_file=core.get_plot_file("preds")
        )
        plot_histogram(fnls, estimates, save_file=core.get_plot_file("hist"))

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = fisher
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)

        data.close()
        save_data(core.file, sdata, mode="a")

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
