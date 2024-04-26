import logging
import os
import sys

import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from ksw.radial_functional import radial_func
from scipy.interpolate import CubicSpline, interp1d
from tqdm.auto import tqdm

from . import Core
from .utils import remove_mono_dipole, save_data, setup_logging
from .utils.plots import plot_cl_alm

logger = setup_logging(__name__, level=logging.DEBUG)


def integrand(alm, bl_div_cl, alpha_l, r, dr, nside, lmax):
    """This calculates the alms from the precalculated values"""
    Balm = hp.almxfl(alm, bl_div_cl, inplace=False)
    B = hp.alm2map(Balm, nside=nside, lmax=lmax, inplace=True)
    inner = hp.map2alm(B**2, lmax=lmax, use_pixel_weights=True)
    kernel = hp.almxfl(inner, alpha_l, inplace=True)
    return dr * r**2 * kernel


def interpolator(func, ells_sparse, axis=1, cubic=True):
    if cubic:
        return CubicSpline(ells_sparse, func, axis)
    else:
        return interp1d(ells_sparse, func, kind="linear", axis=axis)


def generate_alm_ng(s, alms):
    """This code completely calculates, and saves, the alms and almngs."""
    # $$a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]$$
    # and
    # $$\alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)$$
    # $$\beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)$$
    # $$B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}$$
    # where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions
    ells = s.ells[(s.ells >= s.tr_ells.min()) & (s.ells <= s.tr_ells.max())]

    A = (3 / 5) ** 2 * 2 * np.pi**2 * s.cosmo_params["As"]
    delta_phi = (s.tr_k) ** ((s.cosmo_params["ns"] - 4))

    f_k = np.full((len(s.tr_k), 2), 5 / 3, dtype=s.r_dtype)
    # f_k[:, 0] = 1             # f_k for alpha
    f_k[:, 1] *= A * delta_phi  # f_k for beta

    rad = radial_func(f_k, s.tr_ell_k, s.tr_k, s.radii, s.tr_ells)

    alpha_ell = rad[:, :, :, 0]
    alpha_l = interpolator(alpha_ell, s.tr_ells)(ells)

    beta_ell = rad[:, :, :, 1]
    tr_c_ells = s.c_ells[s.tr_ells, : s.npol]
    div = beta_ell / tr_c_ells[np.newaxis, :, :]
    bl_div_cl = interpolator(div, s.tr_ells)(ells)

    logger.info("Starting Alm_ng generation")

    # We use joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    # I also use return_as generator which allows us to consume the results as they are generated (in order),
    # this helps prevent memory issues
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for Alm_ng generation: {temp_folder}")
    parallel = Parallel(n_jobs=-1, return_as="generator", temp_folder=temp_folder)
    alm_ng = np.empty(s.alm_shape, dtype=s.c_dtype)
    for i, pol in tqdm(s.sim_pol, total=s.sim_pol_len, desc="Alm_ng"):
        generator = parallel(
            delayed(integrand)(
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

        # here we consume the generator and sum the results
        alm_ng[i, pol] = sum(generator)
    return alm_ng


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
    s = Core()

    # check for completed alm runs if we are not forcing alm generation, and fail fast
    # notice we don't exit(1) so we can use this to resume a partially completed job
    if not s.force_alm_gen:
        if os.path.isfile(s.alm_file):
            logger.warning("Found completed alms file, skipping alm generation")
            sys.exit(0)
        elif os.path.isfile(s.alm_file_partial):
            logger.warning("Found partial alm file, skipping alm generation")
            sys.exit(0)

    # Get our alms
    logger.info("Starting Alm generation")
    alm_l = np.array(
        [s.data.compute_alm_sim(s.lensing) for _ in range(s.nsims)],  # type: ignore
        dtype=s.c_dtype,
    )
    alm_ng = generate_alm_ng(s, alm_l)
    fnls = s.rng.uniform(s.fnl_min, s.fnl_max, s.fnl_shape)
    alms = alm_l + fnls * alm_ng
    alms = remove_mono_dipole(alms)

    # we remove the stale nc file if it exists
    if os.path.isfile(s.alm_file_partial):
        logger.info("Removing stale alm file: %s", s.alm_file_partial)
        os.remove(s.alm_file_partial)

    logger.info("Saving alm and almng data")
    os.makedirs(s.alm_dir, exist_ok=True)

    sdata = {}
    sdata["alm"] = alms
    sdata["fnl"] = fnls
    save_data(s.alm_file_partial, sdata)

    if s.is_main_job:
        alm_plot_dir = os.path.join(s.plot_dir, "almgen")
        os.makedirs(alm_plot_dir, exist_ok=True)

        i, j = s.rng.integers(s.nsims), s.rng.integers(s.npol)
        logger.debug("Making plots for alm[%d,%d]", i, j)
        filebase = os.path.join(
            alm_plot_dir, f"{s.sjob}_{s.base_name}_alm[{i},{j}].png"
        )

        # plot the complete alms with noise
        plot_cl_alm(
            alms[i, j],
            save_file=filebase,
            plot_camb=True,
            c_ells=s.c_ells,
            camb_noise=True,
            noise=s.noise_ell[0],
            beam_width=s.beam_width,
        )

    logger.info("Finished %s!", s.sjob)
