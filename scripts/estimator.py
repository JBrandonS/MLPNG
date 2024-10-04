import sys
import logging
import os
import healpy as hp
import camb

import h5py
import numpy as np
from mpi4py import MPI

from ksw import KSW, Shape, Cosmology

from . import Core
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
        ret[pol] = hp.almxfl(alm[pol], icov[pol])
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

    # setup our cosmology and compute the c_ells
    camb_params = camb.set_params(**core.cosmo_params, verbose=False)
    cosmo = Cosmology(camb_params)
    cosmo.compute_transfer(core.lmax)
    cosmo.compute_c_ell()

    if core.lensing:
        c_ells = cosmo.c_ell["lensed_scalar"]["c_ell"].T
    else:
        c_ells = cosmo.c_ell["unlensed_scalar"]["c_ell"].T

    loc_shape = Shape.prim_local(
        core.cosmo_params["ns"], core.cosmo_params["pivot_scalar"]
    )
    cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    ksw = KSW(
        cosmo.red_bispectra,
        lambda a: a,  # we will send the alms directly
        core.lmax,
        core.pols,
        core.precision,
    )

    # The default theta_batch size is 25, which is really small, we want to increase it
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    logger.debug("Using theta_batch %s", theta_batch)

    # since we are only doing full sky right now, we do not need to use MC methods
    if core.force_ksw:
        if os.path.exists(core.mc_file):
            logger.info("Loading KSW state from %s", core.mc_file)
            ksw.start_from_read_state(core.mc_file)
        else:
            alm_steps = generate_alm(core, 100)
            logger.debug("Done")

            def step_loader(idx):
                """for stepping the KSW estimator, we just generate new unique sims"""
                logger.debug("Sending alm step %s", idx)
                return alm_steps[idx, : core.npols]

            logger.info("Running KSW step, num steps: %s", 100)
            ksw.step_batch(step_loader, range(100), theta_batch=theta_batch)

            # save the mc state if we are using the mc file
            if True:
                logger.info("Saving KSW state to %s", core.mc_file)
                ksw.write_state(core.mc_file)

        fisher = ksw.compute_fisher()
    else:
        # FIXME
        with np.errstate(divide="ignore", invalid="ignore"):
            ib = np.where(core.beam_ell != 0, 1 / core.beam_ell, 0)
            cl = c_ells
            icov = np.where(
                cl + ib * core.noise_ell * ib != 0,
                1 / (cl + ib * core.noise_ell * ib),
                0,
            )
            icov = icov[core.pol_idxs()]
        fisher = ksw.compute_fisher_isotropic(icov)
    print(f"Fisher: {fisher}, standard deviation: {1 / np.sqrt(fisher)}")

    # FIXME: we need to compute the inverse covariance matrix, support non-iso
    icov_ell = np.zeros_like(c_ells)
    icov_ell = np.where(
        c_ells != 0, 1 / (core.beam_ell**2 * c_ells + core.noise_ell), 0
    )

    # Finally we can get our estimates
    pol_idxs = core.pol_idxs(pretrimmed=True)
    alms = data["alm_lensed"] if core.lensing else data["alm"]  # not in memory yet
    estimates, _, _, _ = ksw.compute_estimate_batch(
        lambda idx: icov_func(icov_ell, alms[idx, pol_idxs]),
        range(core.num_estimates),
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        lin_term=0 if not core.force_ksw else None,
    )

    if mpi_root:
        logger.info("Saving data")

        fnls = np.array(data["fnl"][:])
        data.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = fisher
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(core.file, sdata, mode="a")

        plot_predictions(
            fnls, estimates, fisher=fisher, save_file=core.get_plot_file("preds")
        )
        plot_histogram(fnls, estimates, save_file=core.get_plot_file("hist"))

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
