import logging
import os

import camb
import healpy as hp
import numpy as np
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_root = mpi_rank == 0

from .utils import Config, setup_logging
from .utils.plots import plot_cl_alm, plot_predictions

logger = setup_logging(
    name=f"heidelberg_estimator_{mpi_rank}",
    level=logging.INFO if mpi_root else logging.ERROR,
)

# fix healpy logging because we will get a lot of info
logging.getLogger("healpy").setLevel(logging.WARNING)

# remove astropy warning about verbose that I can't change
logging.getLogger("astropy").setLevel(logging.ERROR)


def alm_loader(str_idx):
    global fnls

    idx = str_idx.zfill(4)
    base1 = f"data/heidelberg/alm_l_{idx}_v3.fits"
    base2 = f"data/heidelberg/alm_nl_{idx}_v3.fits"

    alm_heidelberg_l = np.array(hp.read_alm(base1, hdu=1))
    alm_heidelberg_nl = np.array(hp.read_alm(base2, hdu=1))
    alm_h_l = remove_mono_dipole(alm_heidelberg_l)
    alm_h_nl = remove_mono_dipole(alm_heidelberg_nl)

    fnl = fnls[int(idx)]
    t_scale = 2.7255 * 10 ** (6)
    logger.info("sending fnl: %s", fnl)
    return (alm_h_l + fnl * alm_h_nl) * t_scale


def alm_step_loader(idx):
    # needs a gaussian realization of signal + noise
    return data.compute_alm_sim(s.lensing)


def alm_step_loader2(str_idx):
    idx = str_idx.zfill(4)
    base1 = f"data/heidelberg/alm_l_{idx}_v3.fits"
    alm_heidelberg_l = np.array(hp.read_alm(base1, hdu=1))
    alm_h_l = remove_mono_dipole(alm_heidelberg_l)

    t_scale = 2.7255 * 10 ** (6)
    return alm_h_l * t_scale


def remove_mono_dipole(alm):
    """
    Remove the monopole and dipole terms from the alms.
    """
    lmax = hp.Alm.getlmax(len(alm))
    alm[hp.Alm.getidx(lmax, 0, 0)] = 0.0  # Remove monopole
    alm[hp.Alm.getidx(lmax, 1, 0)] = 0.0  # Remove dipole
    alm[hp.Alm.getidx(lmax, 1, 1)] = 0.0  # Remove dipole
    return alm


def compute_icov_ell(N, b):
    S_ell = cosmo._camb_data.get_cmb_power_spectra(
        cosmo.camb_params,
        s.lmax,
        ["total"],
        "muK",
        True,
    )["total"][:, 0]
    b_inv = 1 / b
    return (1 / (S_ell + b_inv * N * b_inv))[None, :]


def beam(alm):
    return hp.sphtfunc.smoothalm(alm, fwhm=s.beam_width, inplace=False)


if __name__ == "__main__":
    """
    A test script to run the KSW estimator just on the heildelberg sims.
    Probably not up to date with the latest changes in estimator.
    """

    s = Config(["settings/heidelberg.json", "--base_name", "heidelberg"])

    logger.info("Running camb")
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    logger.info("Setting up KSW")
    loc_shape = Shape.prim_local(s.cosmo_params["ns"], s.cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, s.radii)
    data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)
    ksw = KSW(
        cosmo.red_bispectra,
        data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed,
        beam,
        s.lmax,
        s.pols,
        precision=s.precision,
    )

    logger.info("Computing isotropic Fisher")
    icov_ell = compute_icov_ell(s.noise_ell, s.beam_ell)
    fisher_iso = ksw.compute_fisher_isotropic(icov_ell)
    logger.info(
        "Fisher isotropic: %s, std dev: %s", fisher_iso, np.sqrt(1 / fisher_iso)
    )

    logger.info("Generating Fnls")
    fnls = s.rng.uniform(s.fnl_min, s.fnl_max + 1, 1001)

    logger.info("Running KSW step")
    alm_step_strs = np.arange(1, 101).astype(str)
    thetas = int(np.floor(1.5 * s.lmax + 1))
    ksw.step_batch(alm_step_loader, alm_step_strs, comm=mpi_comm, theta_batch=thetas)
    logger.info("Done with step")

    fisher = float(ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    logger.info("Computing estimates")
    alm_strs = range(1, 1001)
    estimates = ksw.compute_estimate_batch(
        alm_loader, alm_strs, comm=mpi_comm, verbose=mpi_root, fisher=fisher
    )

    if mpi_root:
        os.makedirs(s.plot_dir, exist_ok=True)
        pred_file = os.path.join(s.plot_dir, f"{s.sjob}-ksw_heidelberg_predictions.png")
        plot_predictions(fnls[alm_strs], estimates, fisher, save_file=pred_file)

        cl_file = os.path.join(s.plot_dir, f"{s.sjob}-heidelberg_cl.png")
        c_ells = data.cosmology.c_ell["unlensed_scalar"]
        plot_cl_alm(
            alm_loader("1"),
            save_file=cl_file,
            plot_camb=True,
            c_ells=c_ells,
            plot_camb_noise=True,
            noise=s.noise_scale_tt,
            beam=s.beam_width,
        )

    logger.info("Finished %s!", mpi_rank)
