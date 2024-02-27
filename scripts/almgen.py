import logging
import os
import sys

import camb
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from sympy.physics.units import Pa
from ksw import Cosmology, Data
from ksw.radial_functional import radial_func
from numpy.random import randint, uniform
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm
from utils import Config, save_data, setup_logging
from utils.plots import plot_cl_alm
from itertools import product


def remove_mono_dipole(alm):
    """
    Remove the monopole and dipole terms from the alms.
    Note that we do not need -m's due to symmetry
    """
    lmax = hp.Alm.getlmax(len(alm))
    alm[..., hp.Alm.getidx(lmax, 0, 0)] = 0.0  # Remove monopole
    alm[..., hp.Alm.getidx(lmax, 1, 0)] = 0.0  # Remove dipole
    alm[..., hp.Alm.getidx(lmax, 1, 1)] = 0.0  # Remove dipole
    return alm


def get_alm(alm, bl_div_cl, alpha_l, r, dr, nside, lmax):
    """This calculates the alms from the precalculated values"""
    Balm = hp.almxfl(alm, bl_div_cl, inplace=False)
    B = hp.alm2map(Balm, nside=nside, lmax=lmax, pol=False, inplace=False)
    inner = hp.map2alm(B**2, lmax=lmax, pol=False, use_pixel_weights=True)
    kernel = hp.almxfl(inner, alpha_l, inplace=False)
    return dr * r**2 * kernel


def interpolate_ells(func, ells_sparse, ls, axis=1):
    """Our interpolation function to go from space to dense ells."""
    return CubicSpline(ells_sparse, func, axis)(ls)


def generate_almngs(alms):
    """This code completely calculates, and saves, the alms and almngs."""
    # $$a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]$$
    # and
    # $$\alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)$$
    # $$\beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)$$
    # $$B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}$$
    # where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions

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

    logger.info("Starting non-gaussian Alm generation")
    almng = np.zeros((s.nsims, s.npol, s.nelem), dtype=s.c_dtype)
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for almng generation: {temp_folder}")

    for i, pol in tqdm(
        product(range(s.nsims), range(s.npol)), total=s.nsims * s.npol, desc="Almng", miniters=10
    ):

        # create a generator
        alm_gen = Parallel(
            n_jobs=-1, verbose=0, return_as="generator", temp_folder=temp_folder
        )(
            delayed(get_alm)(
                alms[i, pol],
                bl_div_cl[ri, :, pol],
                alpha_l[ri, :, pol],
                s.radii[ri],
                s.drs[ri],
                s.nside,
                s.lmax
            )
            for ri in range(len(s.drs))
        )

        # consume
        for alm in alm_gen:
            almng[i, pol] += alm

    return almng


if __name__ == "__main__":
    r"""
    This code generates the alms
    $$a_{\ell m} = a_{\ell m}^{{G}} + f_{NL}^X a_{\ell m}^{NG}$$
    with
    $a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]$
    and
    $\alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)$
    $\beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)$
    $B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}$
    where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions
    """

    logger = setup_logging("almgen", logging.INFO)
    s = Config(sys.argv[1:])
    is_main = True if s.job_array_index is None or s.job_array_index == 1 else False

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
    ksw_data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)

    # get some data from the KSW data
    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

    # we only need the ells up to lmax
    mask = tr_ells <= s.lmax
    tr_ell_k = tr_ell_k[mask]
    tr_ells = tr_ells[mask]

    # Get our alms
    logger.info("Starting gaussian Alm generation")
    alms = np.array(
        [ksw_data.compute_alm_sim(s.lensing) for _ in range(s.nsims)],
        dtype=s.c_dtype,
    )
    logger.debug("Gaussian Alm shape: %s", alms.shape)

    # KWS needs the alms to be coevolved with the beam
    beam_ell_2d = np.atleast_2d(s.beam_ell)
    for i, j in product(range(s.nsims), range(s.npol)):
        alms[i, j] = hp.almxfl(alms[i, j], beam_ell_2d[j] ** -1)
    logger.info("Gaussian Alm generation complete")

    # get our almngs
    almngs = generate_almngs(alms)
    logger.debug("Non-gaussian Alm shape: %s", almngs.shape)

    # and get the fnls
    fnls = np.random.uniform(s.fnl_min, s.fnl_max+1, (s.nsims, s.npol, 1)).astype(
        s.r_dtype
    )

    # create our combined alms, and remove the monopole and dipole
    complete_alms = alms + fnls * almngs
    complete_alms = remove_mono_dipole(complete_alms)

    sdata = {}
    sdata["alm"] = complete_alms
    sdata["fnl"] = fnls

    logger.debug(f"Completed Alm shape {complete_alms.shape} fnls shape {fnls.shape}")

    if is_main:
        # save the settings if this is the main process
        sdata["settings"] = s.settings

        # polt a random alm and almng for this run
        logger.info("Plotting a random alm and almng")
        i, j = np.random.randint(s.nsims), np.random.randint(s.npol)
        filebase = os.path.join(s.plot_dir, f"{s.sjob}_{s.base_name}_alm[{i},{j}]")
        plot_cl_alm(
            complete_alms[i, j],
            save_file=filebase + f".png",
            plot_camb=True,
            c_ells=c_ells,
        )
        # this is just the gaussian part
        plot_cl_alm(
            alms[i, j],
            save_file=filebase + f"_g.png",
            plot_camb=True,
            c_ells=c_ells,
        )
        # Don't add camb to the ng plots since they are such a small scale
        plot_cl_alm(
            almngs[i, j],
            save_file=filebase + f"_ng.png",
            plot_camb=False,
        )

    logger.info("Saving alm and almng data")
    if not os.path.exists(s.alm_dir):
        logger.info("creating directory: %s", s.alm_dir)
        os.makedirs(s.alm_dir)

    save_data(s.alm_file_nc, sdata)
    os.replace(s.alm_file_nc, s.alm_file_partial)
