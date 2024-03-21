import logging
import os
import sys

import camb
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from ksw import Cosmology, Data
from ksw.radial_functional import radial_func
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm
from utils import Config, save_data, setup_logging
from utils.plots import plot_cl_alm
from itertools import product

logger = setup_logging(__name__)


def remove_mono_dipole(alm):
    """
    Remove the monopole and dipole terms from the alms.
    Note that we do not need -m's due to symmetry
    """
    lmax = hp.Alm.getlmax(len(alm))
    alm[..., hp.Alm.getidx(lmax, 0, 0)] = 0.0
    alm[..., hp.Alm.getidx(lmax, 1, 0)] = 0.0
    alm[..., hp.Alm.getidx(lmax, 1, 1)] = 0.0
    return alm


def get_alm(alm, bl_div_cl, alpha_l, r, dr, nside, lmax):
    """This calculates the alms from the precalculated values"""
    Balm = hp.almxfl(alm, bl_div_cl)
    B = hp.alm2map(Balm, nside=nside, lmax=lmax)
    inner = hp.map2alm(B**2, lmax=lmax, use_pixel_weights=True)
    kernel = hp.almxfl(inner, alpha_l)
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

    # get some data from the KSW data
    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

    # we only need the ells up to lmax
    mask = tr_ells <= s.lmax
    tr_ell_k = tr_ell_k[mask]
    tr_ells = tr_ells[mask]

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

    # We use joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for almng generation: {temp_folder}")
    parallel = Parallel(n_jobs=-1, return_as="generator", temp_folder=temp_folder)
    for i, pol in tqdm(
        product(range(s.nsims), range(s.npol)), total=(s.nsims * s.npol), desc="Almng"
    ):
        logger.debug("Generating almng[%d,%d]", i, pol)
        generator = parallel(
            delayed(get_alm)(
                alms[i, pol],
                bl_div_cl[ri, :, pol],
                alpha_l[ri, :, pol],
                s.radii[ri],
                s.drs[ri],
                s.nside,
                s.lmax,
            )
            for ri in range(len(s.drs))
        )
        logger.debug("Summing almng[%d,%d]", i, pol)
        almng[i, pol] = sum(generator)
        logger.debug("Almng[%d,%d] shape: %s", i, pol, almng[i, pol].shape)
    logger.info("Non-gaussian Alm generation complete")
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
    s = Config()

    if s.is_main_job:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("utils.config").setLevel(logging.DEBUG)
        logging.getLogger("utils.utils").setLevel(logging.DEBUG)

    # check for completed alm runs if we are not forcing alm generation, and fail fast
    if not s.force_alm_gen:
        if os.path.isfile(s.alm_file):
            logger.warning("Found completed alms file, skipping alm generation")
            sys.exit(0)
        elif os.path.isfile(s.alm_file_partial):
            # this allows us to stop and start the generation
            logger.warning(f"Found partial alm file, skipping alm generation")
            sys.exit(0)

    # here we setup camb
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    # setup the KSW data
    ksw_data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)

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
    fnls = s.rng.uniform(s.fnl_min, s.fnl_max + 1, (s.nsims, s.npol, 1))

    # create our combined alms, and remove the monopole and dipole
    complete_alms = alms + fnls * almngs
    complete_alms = remove_mono_dipole(complete_alms)

    # we remove the stale nc file if it exists
    if os.path.isfile(s.alm_file_partial):
        logger.info("Removing stale alm file: %s", s.alm_file_partial)
        os.remove(s.alm_file_partial)

    logger.info("Saving alm and almng data")
    os.makedirs(s.alm_dir, exist_ok=True)

    sdata = {}
    sdata["alm"] = complete_alms
    sdata["fnl"] = fnls
    # TODO: Fix this
    # if s.is_main_job:
    #     logger.info('adding settings to alm file')
    #     import json
    #     sdata["settings"] = json.dumps(s.settings)
    save_data(s.alm_file_partial, sdata)

    if s.is_main_job:
        alm_plot_dir = os.path.join(s.plot_dir, "almgen")
        os.makedirs(alm_plot_dir, exist_ok=True)

        c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]

        i, j = s.rng.integers(s.nsims), s.rng.integers(s.npol)
        logger.info("Making plots for alm[%d,%d]", i, j)
        filebase = os.path.join(alm_plot_dir, f"{s.sjob}_{s.base_name}_alm[{i},{j}]")

        # plot the complete alms with noise
        plot_cl_alm(
            complete_alms[i, j],
            save_file=f"{filebase}.png",
            plot_camb=True,
            c_ells=c_ells,
            plot_camb_noise=s.noise,
            noise=s.noise_scale_tt,
            beam=s.beam_width,
        )

        # plot just the gaussian part
        plot_cl_alm(
            alms[i, j],
            save_file=f"{filebase}_g.png",
            plot_camb=True,
            c_ells=c_ells,
            plot_camb_noise=s.noise,
            noise=s.noise_scale_tt,
            beam=s.beam_width,
        )

        # plot the ng part, we don't add camb to the plots since they are such a small scale
        plot_cl_alm(almngs[i, j], save_file=f"{filebase}_ng.png")

    logger.info("Finished %s!", s.sjob)
