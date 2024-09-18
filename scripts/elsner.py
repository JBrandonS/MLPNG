import sys
import logging
import os

import healpy as hp
import numpy as np
from mpi4py import MPI
from ksw import KSW, Shape

from . import Core
from .utils import remove_mono_dipole, setup_logging, trim_alms, print_errors
from .utils.plots import plot_cl_alm, plot_predictions, plot_cl

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

    logger.debug("Checking for files and will download if needed")

    URL0 = "http://dc.zah.uni-heidelberg.de/elsnersim/q/s/static/cl_wmap5_bao_sn.txt"
    file0 = "data/elsner/cl_wmap5_bao_sn.dat"
    if not os.path.exists(file0):
        with requests.Session() as s, open(file0, "wb") as f:
            f.write(s.get(URL0).content)

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

def icov_func(beam, noise, npol, datafile="data/elsner/cl_wmap5_bao_sn.dat"):
    """
    Returned the inverse covariance of the data, used in the KSW estimator.

    Function takes (npol, nelem) alm-like complex array "alm" and returns the
    inverse-variance-weighted version of that array. Specifically:
    (B^{-1} N B^{-1} + S)^{-1} B^{-1} a, where a = B s + n, B is the beam
    and N^{-1} and S^{-1} are the inverse noise and signal covariance
    matrices, respectively.

    Parameters:
    - alm (ndarray): (npol, nelem) alm-like complex array.

    Returns:
    - ndarray: Inverse-variance-weighted version of the input array.
    """

    B_inv = 1 / beam

    # is in in shape (ell, cat) with cat=l, TT, EE, TE
    data = np.loadtxt(datafile)
    data = data.transpose()
    lmax = int(data[0, -1])
    S = np.zeros((npol, lmax + 1))
    S[:, 2:] = data[1 : npol + 1]

    # convert from dimensionless to muK^2
    S *= (2.7255 * 1e6) ** 2

    factor = (B_inv * noise * B_inv + S) ** (-1) * B_inv

    def _func(alm):
        ret = np.zeros_like(alm)
        for pol in range(npol):
            ret[pol] = hp.almxfl(alm[pol], factor[pol])
        return ret

    return _func


def main():
    """
    A test script to run the KSW estimator just on the elsner sims.
    Probably not up to date with the latest changes in estimator.
    """
    get_files()
    mpi_comm.Barrier()

    core = Core(["settings/elsner.json", "--base_name", "elsner_test"] + sys.argv[1:])
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, (1000, 1))

    loc_shape = Shape.prim_local(
        core.cosmo_params["ns"], core.cosmo_params["pivot_scalar"]
    )
    core.cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    ksw = KSW(
        core.cosmo.red_bispectra,
        None,
        core.lmax,
        core.pols,
        core.precision,
    )

    # The default theta_batch size is 25, which is really small, we want to increase it
    # going too high can cause memory issues, so we will cap it at 256
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    theta_batch = min(8, theta_batch)

    fisher = float(ksw.compute_fisher_isotropic())
    logger.debug("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    def alm_loader(str_idx):
        idx = str(str_idx).zfill(4)
        base1 = f"data/elsner/alm_l_{idx}_v3.fits"
        base2 = f"data/elsner/alm_nl_{idx}_v3.fits"

        hdu = 1 if not core.use_pols else (1, 2)
        alm_elsner_l = np.array(hp.read_alm(base1, hdu))
        alm_elsner_nl = np.array(hp.read_alm(base2, hdu))

        fnl = fnls[int(str_idx) - 1]
        t_scale = core.cosmo.camb_params.TCMB * 1e6
        logger.debug("Sending fnl: %s for step %s", fnl, str_idx)

        alms = (alm_elsner_l + fnl[..., np.newaxis] * alm_elsner_nl) * t_scale
        alms = alms[: core.npols]

        if mpi_root and str_idx == 1:
            plot_dir = os.path.join(core.plot_dir, "elsner_test")
            os.makedirs(plot_dir, exist_ok=True)

            cl_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}_cl.png")
            c_ells = core.cosmo.c_ell["unlensed_scalar"]["c_ell"]
            c_ells = np.transpose(c_ells)
            plot_cl_alm(alms, save_file=cl_file, plot_camb=True, camb_cls=c_ells)

        # need sto trim the values if we are using a lower lmax, throw error if asking for higher lmax
        lmax = hp.Alm.getlmax(alms.shape[-1])
        if lmax < core.lmax:
            # this could be done earlier but this is just test code so not super important to optimize
            raise ValueError(
                "alm has lmax %s < %s which cannot be resolved", lmax, core.lmax
            )
        if lmax > core.lmax:
            alms = trim_alms(alms, core.lmax)
        return alms

    logger.info("Computing estimates")
    estimates, _, _, _ = ksw.compute_estimate_batch(
        alm_loader, range(1, 1001), comm=mpi_comm, fisher=fisher, lin_term=0
    )

    if mpi_root:
        plot_dir = os.path.join(core.plot_dir, "elsner_test")
        os.makedirs(plot_dir, exist_ok=True)

        pred_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}_preds.png")
        plot_predictions(fnls[:-1], estimates, fisher=fisher, save_file=pred_file)

        cl_file = os.path.join(plot_dir, f"{core.sjob}-{core.base_name}_cl.png")
        c_ells = core.cosmo.c_ell["unlensed_scalar"]["c_ell"]
        c_ells = np.transpose(c_ells)
        plot_cl_alm(alm_loader("1"), save_file=cl_file, plot_camb=True, camb_cls=c_ells)

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
