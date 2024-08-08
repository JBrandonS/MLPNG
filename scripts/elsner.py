import sys
import logging
import os

import healpy as hp
import numpy as np
from mpi4py import MPI
import camb
from ksw import KSW, Cosmology, Shape

from . import Core
from .estimator import icov_func, conv_beam_func, run_ksw_step
from .utils import remove_mono_dipole, setup_logging, trim_alms, print_errors
from .utils.plots import plot_cl_alm, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

logger = setup_logging(
    name=f"elsner_estimator_{mpi_rank}",
    level=logging.DEBUG if mpi_root else logging.ERROR,
)


def get_files():
    import requests

    logger.info("Checking for files and will download if needed")
    for num in range(mpi_rank + 1, 1001, mpi_size):
        URL1 = (
            "https://dc.zah.uni-heidelberg.de/elsnersim/q/s/static/alm_l_"
            + str(num).zfill(4)
            + "_v3.fits"
        )
        URL2 = (
            "https://dc.zah.uni-heidelberg.de/elsnersim/q/s/static/alm_nl_"
            + str(num).zfill(4)
            + "_v3.fits"
        )

        file1 = "data/elsner/alm_l_" + str(num).zfill(4) + "_v3.fits"
        if not os.path.exists(file1):
            with requests.Session() as s, open(file1, "wb") as f:
                f.write(s.get(URL1).content)

        file2 = "data/elsner/alm_nl_" + str(num).zfill(4) + "_v3.fits"
        if not os.path.exists(file2):
            with requests.Session() as s, open(file2, "wb") as f:
                f.write(s.get(URL2).content)


def main():
    """
    A test script to run the KSW estimator just on the heildelberg sims.
    Probably not up to date with the latest changes in estimator.
    """
    get_files()
    mpi_comm.Barrier()

    core = Core(["settings/elsner.json", "--base_name", "elsner_test"] + sys.argv[1:])
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max + 1, (1000, 1))

    # we need to setup the KSW here
    cosmo_params = core.cosmo_params
    cosmo = Cosmology(camb.set_params(**cosmo_params))
    cosmo.compute_transfer(core.max_l)
    cosmo.compute_c_ell()

    noise = core.noise_ell[: core.npol]
    beam = core.beam_ell[: core.npol]
    c_ells = core.c_ells[: core.npol]

    loc_shape = Shape.prim_local(cosmo_params["ns"], cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    # data = Data(core.lmax, noise, beam, core.pols, cosmo)

    ksw = KSW(
        cosmo.red_bispectra,
        icov_func(beam, noise, core.c_ells, core.npol),
        conv_beam_func(core),
        core.lmax,
        core.pols,
        core.precision,
    )

    # The default theta_batch size is 25, which is really small, we want to increase it
    # going too high can cause memory issues, so we will cap it at 256
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    theta_batch = min(256, theta_batch)

    logger.info("Running KSW step")
    run_ksw_step(ksw, core, theta_batch)
    logger.info("Done with step")

    fisher = float(ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    def alm_loader(str_idx):
        idx = str(str_idx).zfill(4)
        base1 = f"data/elsner/alm_l_{idx}_v3.fits"
        base2 = f"data/elsner/alm_nl_{idx}_v3.fits"

        hdu = 1 if not core.use_pols else (1, 2)
        alm_elsner_l = np.array(hp.read_alm(base1, hdu))
        alm_elsner_nl = np.array(hp.read_alm(base2, hdu))

        fnl = fnls[int(str_idx) - 1]
        t_scale = 2.7255 * 10 ** (6)
        logger.debug("Sending fnl: %s", fnl)

        alms = (alm_elsner_l + fnl[..., np.newaxis] * alm_elsner_nl) * t_scale
        alms = remove_mono_dipole(alms[: core.npol])

        # need sto trim the values if we are using a lower lmax, throw error if asking for higher lmax
        lmax = hp.Alm.getlmax(alms.shape[-1])
        if lmax < core.lmax:
            # this could be done earlyer but this is just test code so not super important to optimize
            raise ValueError(
                "alm has lmax %s < %s which cannot be resolved", lmax, core.lmax
            )
        if lmax > core.lmax:
            logger.debug("Trimming alm from lmax %s to %s", lmax, core.lmax)
            alms = trim_alms(alms, core.lmax)
        return alms

    logger.info("Computing estimates")
    alm_strs = range(1, 1001)
    estimates = ksw.compute_estimate_batch(
        alm_loader, alm_strs, comm=mpi_comm, fisher=fisher
    )

    if mpi_root:
        plot_dir = os.path.join(core.plot_dir, "elsner_test")
        os.makedirs(plot_dir, exist_ok=True)

        pred_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}.png")
        plot_predictions(fnls, estimates, fisher=fisher, save_file=pred_file)

        cl_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}_cl.png")
        c_ells = cosmo.c_ell["unlensed_scalar"]["c_ell"]
        c_ells = np.transpose(c_ells)
        plot_cl_alm(alm_loader("1"), save_file=cl_file, plot_camb=True, camb_cls=c_ells)
        print_errors(fnls, estimates, fisher)

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
