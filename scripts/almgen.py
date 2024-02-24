import logging
import os
import sys

import camb
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from ksw import Cosmology, Data
from ksw.radial_functional import radial_func
from numpy.random import randint, uniform
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm
from utils import Config, save_data, setup_logging
from utils.plots import plot_cl_alm


def get_alm(alm, bl_div_cl, alpha_l, r, dr):
    """This calculates the alms from the precalculated values"""
    Balm = hp.almxfl(alm, bl_div_cl, inplace=False)
    B = hp.alm2map(Balm, nside=s.nside, lmax=s.lmax, pol=False, inplace=False)
    inner = hp.map2alm(B**2, lmax=s.lmax, pol=False, use_pixel_weights=True)
    kernel = hp.almxfl(inner, alpha_l, inplace=False)
    return dr * r**2 * kernel


def interpolate_ells(func, ells_sparse, ls, axis=1):
    """Our interpolation function to go from space to dense ells."""
    return CubicSpline(ells_sparse, func, axis)(ls)


def generate_almngs(plot=False):
    """This code completely calculates, and saves, the alms and almngs."""
    A = (3 / 5) ** 2 * 2 * np.pi**2 * s.cosmo_params["As"]
    delta_phi = (tr_k) ** ((s.cosmo_params["ns"] - 1)) / (tr_k**3)

    f_k = np.ones((len(tr_k), 2), dtype=s.r_dtype) * 5 / 3
    # f_k[:, 0] = 1             # f_k for alpha
    f_k[:, 1] *= A * delta_phi  # f_k for beta

    rad = radial_func(f_k, tr_ell_k, tr_k, s.radii, tr_ells)

    alpha_ell = rad[:, :, :, 0]
    alpha_l = np.concatenate(np.array([interpolate_ells(alpha_ell, tr_ells, s.ells)]))
    alpha_l = np.ascontiguousarray(alpha_l)

    beta_ell = rad[:, :, :, 1]
    c_ells_new = c_ells["c_ell"][tr_ells, : s.npol]
    div = beta_ell / c_ells_new[np.newaxis, :, :]

    bl_div_cl = np.concatenate(np.array([interpolate_ells(div, tr_ells, s.ells)]))
    bl_div_cl = np.ascontiguousarray(bl_div_cl)

    # Each alm takes ~30Mb at 1024. This is fast enough we don't need to parallelize even for very large datasets
    logger.info("Starting gaussian Alm generation")
    alms = np.array(
        [
            ksw_data.compute_alm_sim(s.lensing)
            for _ in tqdm(range(s.nsims), desc="Alm progress")
        ],
        dtype=s.c_dtype,
    )

    logger.info("Applying beam to alms")
    # KSW expects the alms to be coevolved with the beam
    # Make sure we dont get error from the beam_ell being a vector
    beam_ell_2d = np.atleast_2d(beam_ell)
    for i in range(s.nsims):
        for j in range(s.npol):
            alms[i, j] = hp.almxfl(alms[i, j], beam_ell_2d[j] ** -1)

    logger.info("Starting non-gaussian Alm generation")
    sim_data = np.zeros((s.nsims, s.npol, s.nelem), dtype=s.c_dtype)
    for i in tqdm(range(s.nsims), desc="NG Alm progress"):
        for pol in range(s.npol):
            # create a generator
            alm_gen = Parallel(n_jobs=-1, verbose=0, return_as="generator")(
                delayed(get_alm)(
                    alms[i, pol],
                    bl_div_cl[ri, :, pol],
                    alpha_l[ri, :, pol],
                    s.radii[ri],
                    s.drs[ri],
                )
                for ri in range(len(s.drs))
            )

            # consume
            for alm in alm_gen:
                sim_data[i, pol] += alm

    logger.info("Done!")

    logger.info("Saving alm and almng data")
    sdata = {}
    sdata["alm"] = alms
    sdata["almng"] = sim_data
    if is_main:
        # save the settings if this is the main process
        sdata["settings"] = s.settings

        if plot:
            # polt a random alm and almng for this run
            i, j = np.random.randint(s.nsims), np.random.randint(s.npol)
            alm_plot = os.path.join(s.plot_dir, s.base_name + f"_alm[{i},{j}].png")
            almng_plot = os.path.join(s.plot_dir, s.base_name + f"_almng[{i},{j}].png")

            logger.info("Plotting alm and almng")
            plot_cl_alm(alms[i, j], save_file=alm_plot, plot_camb=True, c_ells=c_ells)
            # Don't add camb to the ng plots since they are a much small scale
            plot_cl_alm(sim_data[i, j], save_file=almng_plot, plot_camb=False)

    save_data(s.alm_file_nc, sdata)
    os.replace(s.alm_file_nc, s.alm_file_partial)


if __name__ == "__main__":
    # This code generates the alms
    # $$a_{\ell m} = a_{\ell m}^{{G}} + f_{NL}^X a_{\ell m}^{NG}$$
    # Most of this code is to calculate the term (eq. 27)
    # $$a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]$$
    # and
    # $$\alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)$$
    # $$\beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)$$
    # $$B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}$$
    # where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions

    s = Config(sys.argv[1:])
    is_main = True if s.job_array_index is None or s.job_array_index == 1 else False
    logger = setup_logging("almgen", logging.INFO if is_main else logging.ERROR)

    # disable some info logs from healpy that clutter the output with
    # healpy - INFO - Sigma is 0.000000 arcmin (0.000000 rad)
    # healpy - INFO - -> fwhm is 0.000000 arcmin
    # logging.getLogger("healpy").setLevel(logging.ERROR)

    # check for completed alm runs if we are not forcing alm generation
    if not s.force_alm_gen:
        if os.path.isfile(s.alm_file):
            logger.info("Found completed alms file, skipping alm generation")
            exit(0)
        elif os.path.isfile(s.alm_file_partial):
            logger.info(f"Found partial alm file, skipping alm generation")
            exit(0)

    # we remove the stale nc file if it exists
    if os.path.isfile(s.alm_file_nc):
        logger.info("Removing stale alm file: %s", s.alm_file_nc)
        os.remove(s.alm_file_nc)

    # here we setup camb
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    # setup the KSW data
    noise_ell, beam_ell = s.noise_beam
    ksw_data = Data(s.lmax, noise_ell, beam_ell, s.pols, cosmo)

    # get some data from the KSW data
    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

    # we only need the ells up to lmax
    mask = tr_ells <= s.lmax
    tr_ell_k = tr_ell_k[mask]
    tr_ells = tr_ells[mask]

    generate_almngs(plot=is_main)
