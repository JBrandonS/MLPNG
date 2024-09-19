import sys
import logging
import os
import healpy as hp

import h5py
import numpy as np
from mpi4py import MPI

from . import Core
from .generator import generate_alm
from .utils import save_data, setup_logging, print_errors
from .utils.plots import plot_histogram, plot_predictions

from ksw import KSW, Shape

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


def icov_func(core, alm):
    ret = np.zeros_like(alm)
    for pol in range(ret.shape[0]):
        ret[pol] = hp.almxfl(alm[pol], core.icov_tot[pol])
    return ret


def run_ksw_step(ksw, core, theta_batch, num_steps=100):
    alm_steps = generate_alm(core, num_steps)

    def step_loader(idx):
        """for stepping the KSW estimator, we just generate new unique sims"""
        logger.debug("Sending alm step %s", idx)
        return alm_steps[idx]

    ksw.step_batch(step_loader, range(num_steps), mpi_comm, theta_batch=theta_batch)


def main():
    core = Core()

    # the KSW code requires the total_sims to be >= mpi_size
    # best usage would have total_sims % mpi_size == 0, but not required
    assert (
        core.total_sims >= mpi_size
    ), "total_sims < mpi_size, lower ntasks or increase sims"

    # early loading to fail fast if the file does not exist
    data_file = h5py.File(core.file_complete, "r", swmr=True, locking=False)

    loc_shape = Shape.prim_local(
        core.cosmo_params["ns"], core.cosmo_params["pivot_scalar"]
    )
    core.cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    ksw = KSW(
        core.cosmo.red_bispectra,
        lambda a: a,
        core.lmax,
        core.pols,
        core.precision,
    )

    # The default theta_batch size is 25, which is really small, we want to increase it
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    logger.debug("Using theta_batch %s", theta_batch)

    # since we are only doing full sky right now, we do not need to use MC methods
    if core.force_ksw:  # TODO: test mc methods
        if os.path.exists(core.mc_file):
            logger.info("Loading KSW state from %s", core.mc_file)
            ksw.start_from_read_state(core.mc_file, mpi_comm)
        else:  # not tested much
            run_ksw_step(ksw, core, theta_batch)

            # save the mc state if we are using the mc file
            if mpi_root:
                logger.info("Saving KSW state to %s", core.mc_file)
                ksw.write_state(core.mc_file, mpi_comm)

        fisher = float(ksw.compute_fisher())  # type: ignore
    else:
        icov_ell = core.icov_tot[core.pol_idxs()]
        fisher = ksw.compute_fisher_isotropic(icov_ell, comm=mpi_comm)
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory, yet
    alms = data_file["alm_lensed"] if core.lensing else data_file["alm"]

    # get the polarization indexes but do not yet trim the alms as they will be loaded into memory
    pol_idxs = core.pol_idxs(pretrimmed=True)

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

    def alm_loader(idx):
        """load the alms into memory with debug logging"""
        logger.debug("Sending alm %s", idx)
        return icov_func(core, alms[idx, pol_idxs])

    estimates, cubic_terms, lin_terms, fisher_terms = ksw.compute_estimate_batch(
        alm_loader,
        range(core.num_estimates),
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        lin_term=0 if not core.force_ksw else None,
    )

    if mpi_root:
        logger.info("Saving data")

        fnls = np.array(data_file["fnl"][:]).flatten()
        data_file.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = np.atleast_1d(fisher)
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(core.file_complete, sdata, mode="a")

        # and lets make some plots
        plot_dir = os.path.join(core.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)

        pred_file = os.path.join(plot_dir, f"{core.base_name}_{core.sjob}_preds.png")
        plot_predictions(fnls, estimates, fisher=fisher, save_file=pred_file)

        hist_file = os.path.join(plot_dir, f"{core.base_name}_{core.sjob}_hist.png")
        plot_histogram(fnls, estimates, save_file=hist_file)
        print_errors(fnls, estimates, fisher)

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
