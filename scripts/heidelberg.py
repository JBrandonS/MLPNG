import sys
import logging
import os

import healpy as hp
import numpy as np
from mpi4py import MPI

from . import Core
from .utils import remove_mono_dipole, setup_logging
from .utils.plots import plot_cl_alm, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_root = mpi_rank == 0

logger = setup_logging(
    name=f"heidelberg_estimator_{mpi_rank}",
    level=logging.INFO if mpi_root else logging.ERROR,
)


def alm_loader(str_idx):
    global fnls

    idx = str(str_idx).zfill(4)
    base1 = f"data/heidelberg/alm_l_{idx}_v3.fits"
    base2 = f"data/heidelberg/alm_nl_{idx}_v3.fits"

    alm_heidelberg_l = np.array(hp.read_alm(base1, hdu=1))
    alm_heidelberg_nl = np.array(hp.read_alm(base2, hdu=1))

    fnl = fnls[int(idx)]
    t_scale = 2.7255 * 10 ** (6)
    logger.info("sending fnl: %s", fnl)

    alms = (alm_heidelberg_l + fnl * alm_heidelberg_nl) * t_scale
    alms = remove_mono_dipole(alms)
    return alms


if __name__ == "__main__":
    """
    A test script to run the KSW estimator just on the heildelberg sims.
    Probably not up to date with the latest changes in estimator.
    """

    args = sys.argv[1:]
    core = Core(["settings/heidelberg.json"] + args)

    def alm_step_loader(idx):
        return core.ksw.compute_alm_sim(core.lensing)

    logger.info("Generating Fnls")
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max + 1, 1001)

    logger.info("Running KSW step")
    alm_step_strs = np.arange(1, 101).astype(str)
    thetas = int(np.floor(1.5 * core.lmax + 1))
    core.ksw.step_batch(
        alm_step_loader, alm_step_strs, comm=mpi_comm, theta_batch=thetas
    )
    logger.info("Done with step")

    fisher = float(core.ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    logger.info("Computing estimates")
    alm_strs = range(1, 1001)
    estimates = core.ksw.compute_estimate_batch(
        alm_loader, alm_strs, comm=mpi_comm, fisher=fisher
    )

    if mpi_root:
        plot_dir = os.path.join(core.plot_dir, "heidelberg_test")
        os.makedirs(plot_dir, exist_ok=True)
        
        pred_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}.png")
        plot_predictions(fnls[alm_strs], estimates, fisher=fisher, save_file=pred_file)

        cl_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}_cl.png")
        c_ells = core.cosmo.c_ell["unlensed_scalar"]
        plot_cl_alm(alm_loader("1"), save_file=cl_file, plot_camb=True, c_ells=c_ells)

    logger.info("Finished %s!", mpi_rank)
