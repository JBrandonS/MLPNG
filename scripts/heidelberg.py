import logging
import os
import sys

import camb
import healpy as hp
import numpy as np
from astropy import units as u
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI

from utils import Config, setup_logging
from utils.plots import plot_ksw_predictions, plot_cl_alm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_main_comm = rank == 0

# fix healpy logging because we will get a lot of info
logging.getLogger("healpy").setLevel(logging.WARNING)

# remove astropy warning about verbose that I can't change
logging.getLogger("astropy").setLevel(logging.ERROR)


def alm_loader(str_idx):
    global fnls

    idx = str_idx.zfill(4)
    t_scale = 2.7255 * 10 ** (6)

    base1 = f"data/heidelberg/alm_l_{idx}_v3.fits"
    base2 = f"data/heidelberg/alm_nl_{idx}_v3.fits"

    alm_heidelberg_l = np.array(hp.read_alm(base1, hdu=1))
    alm_heidelberg_nl = np.array(hp.read_alm(base2, hdu=1))

    alm_h_l = remove_mono_dipole(alm_heidelberg_l)
    alm_h_nl = remove_mono_dipole(alm_heidelberg_nl)

    # rng = np.random.default_rng()
    fnl = fnls[int(idx)]
    logging.info("sending fnl: %s", fnl)
    return (alm_h_l + fnl * alm_h_nl) * t_scale


def alm_step_loader(idx):
    # needs a gaussian realization of signal + noise
    return data.compute_alm_sim(s.lensing)


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
        cosmo.camb_params, s.lmax, ["total"], "muK", True
    )["total"][:, 0]
    b_inv = 1 / b
    return (1 / (S_ell + b_inv * N * b_inv))[None, :]


if __name__ == "__main__":
    """
    A test script to run the KSW estimator just on the heildelberg sims.
    Probably not up to date with the latest changes in estimator.
    """

    logger = setup_logging(
        name=f"heidelberg_estimator_{rank}",
        level=logging.INFO if is_main_comm else logging.ERROR,
    )
    s = Config(["settings/heidelberg.json"])

    s.fnl_min = -50
    s.fnl_max = 50
    fnls = np.random.uniform(s.fnl_min, s.fnl_max, s.total_sims)

    # init camb and setup the reduced bispecturm to local
    logger.info("Running camb")
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()
    logger.info("done starting camb")

    # create the local shape
    loc_shape = Shape.prim_local(s.cosmo_params["ns"], s.cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, s.radii)

    # setup the data and get our icov object
    data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)
    icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

    # generate our beam functioned based on noise
    beam_width_rad = 0 if s.disable_noise else s.beam_width.to_value(u.radian)

    def beam(alm):
        if beam_width_rad == 0:
            return alm

        return hp.sphtfunc.smoothalm(alm, fwhm=beam_width_rad, inplace=False)

    ksw = KSW(
        cosmo.red_bispectra,
        icov,
        beam,
        s.lmax,
        s.pols,
        precision="double" if s.double_precision else "single",
    )

    alm_strs = np.arange(1, s.total_sims + 1).astype(str)
    alm_step_strs = np.arange(1, 100) if s.total_sims > 100 else alm_strs

    logger.info("Running KSW step")
    ksw.step_batch(alm_step_loader, alm_step_strs, comm)
    logger.info("Done with KSW step")

    logger.info("Computing estimates")
    fisher = ksw.compute_fisher()
    estimates = ksw.compute_estimate_batch(
        alm_loader, alm_strs, comm, verbose=(rank == 0), fisher=fisher
    )
    logger.info("done")

    # compute isotropic fisher
    icov_ell = compute_icov_ell(s.noise_ell, s.beam_ell)
    fisher_iso = ksw.compute_fisher_isotropic(icov_ell, comm=comm)

    logger.info("Finished %s!", rank)

    if is_main_comm:
        fnls = np.array(fnls[alm_strs.astype(int)])

        pred_file = os.path.join(s.plot_dir, "ksw_heidelberg_predictions.png")
        plot_ksw_predictions(fnls, estimates, fisher, save_file=pred_file)

        cl_file = os.path.join(s.plot_dir, "heidelberg_cl.png")
        c_ells = data.cosmology.c_ell["unlensed_scalar"]
        plot_cl_alm(
            alm_loader("1"),
            save_file=cl_file,
            plot_camb=True,
            c_ells=c_ells,
            plot_camb_noise=True,
            noise_scale_tt=s.noise_scale_tt,
            beam_width=s.beam_width,
        )
