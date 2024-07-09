import sys
import logging
import os

import healpy as hp
import numpy as np
from mpi4py import MPI
import camb
from ksw import KSW, Cosmology, Shape

from . import Core
from .generator import generate_alm
from .estimator import icov_func, conv_beam_func
from .utils import remove_mono_dipole, setup_logging
from .utils.plots import plot_cl_alm, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

logger = setup_logging(
    name=f"heidelberg_estimator_{mpi_rank}",
    level=logging.INFO if mpi_root else logging.ERROR,
)


def main():
    """
    A test script to run the KSW estimator just on the heildelberg sims.
    Probably not up to date with the latest changes in estimator.
    """

    args = sys.argv[1:]
    core = Core(["settings/heidelberg.json"] + args)

    logger.info("Generating Fnls")
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max + 1, 1001)

    # The default theta_batch size is 25, which is really small, we want to increase it
    # going too high can cause memory issues, so we will cap it at 256
    thetas = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    thetas = min(256, int(np.floor(1.5 * core.lmax + 1)))

    # we need to setup the KSW here
    cosmo_params = core.cosmo_params
    cosmo = Cosmology(camb.set_params(**cosmo_params))
    cosmo.compute_transfer(core.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    pols = tuple([pol for pol in core.pols if pol != "B"])  # remove the b-modes
    npols = len(pols)
    nnoise = 1 if npols == 1 else 3

    noise = core.noise_ell[:nnoise]
    beam = core.beam_ell[:npols]

    loc_shape = Shape.prim_local(cosmo_params["ns"], cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    icov = icov_func(beam, noise[:npols], core.c_ells[:npols])

    ksw = KSW(
        cosmo.red_bispectra,
        icov,
        conv_beam_func(core, npols),
        core.lmax,
        pols,
        core.precision,
    )

    logger.info("Running KSW step")
    step_size = 100
    alm_step_strs = np.arange(step_size)
    step_alms = generate_alm(core, step_size)

    def alm_step_loader(idx):
        return step_alms[idx]

    ksw.step_batch(alm_step_loader, alm_step_strs, comm=mpi_comm, theta_batch=thetas)
    logger.info("Done with step")

    fisher = float(ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    def alm_loader(str_idx):
        idx = str(str_idx).zfill(4)
        base1 = f"data/heidelberg/alm_l_{idx}_v3.fits"
        base2 = f"data/heidelberg/alm_nl_{idx}_v3.fits"

        alm_heidelberg_l = np.array(hp.read_alm(base1, hdu=1))
        alm_heidelberg_nl = np.array(hp.read_alm(base2, hdu=1))

        fnl = fnls[int(idx)]
        t_scale = 2.7255 * 10 ** (6)
        logger.debug("sending fnl: %s", fnl)

        alms = (alm_heidelberg_l + fnl * alm_heidelberg_nl) * t_scale
        alms = remove_mono_dipole(alms)
        return alms

    logger.info("Computing estimates")
    alm_strs = range(1, 1001)
    estimates = ksw.compute_estimate_batch(
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


if __name__ == "__main__":
    sys.exit(main())
