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


def integrand(alm, bl_div_cl, alpha_l, nside, lmax, radii):
    """
    Calculate the integrand for a given set of parameters.

    Parameters:
    alm (ndarray): The input spherical harmonic coefficients.
    bl_div_cl (ndarray): The division of beam transfer function and C_l.
    alpha_l (ndarray): The alpha_l coefficients.
    r (float): The radial distance.
    dr (float): The differential radial distance.
    nside (int): The HEALPix nside parameter.
    lmax (int): The maximum multipole moment.

    Returns:
    ndarray: The calculated integrand.

    """

    Balm = hp.almxfl(alm, bl_div_cl)
    # logger.info("Balm shape %s", Balm.shape)
    B = hp.alm2map(Balm, nside=nside)
    # logger.info("B shape %s", B.shape)
    inner = hp.map2alm(B**2, lmax=lmax, use_pixel_weights=True)
    # logger.info("inner shape %s", inner.shape)
    return radii**2 * hp.almxfl(inner, alpha_l)


def interpolator(func, ells_sparse, axis=1, cubic=False):
    """
    Interpolates a function using either cubic spline or linear interpolation.

    Parameters:
        func (array_like): The function values to be interpolated.
        ells_sparse (array_like): The sparse grid points.
        axis (int, optional): The axis along which to interpolate. Default is 1.
        cubic (bool, optional): If True, cubic spline interpolation is used. If False, linear interpolation is used. Default is True.

    Returns:
        object: An interpolating function.

    """
    if cubic:
        return CubicSpline(ells_sparse, func, axis)
    else:
        return interp1d(ells_sparse, func, kind="linear", axis=axis)


def generate_alm_ng(core, alms):
    """
    This function calculates the non-gaussian alms using the given core and the gaussian alms.

    Parameters:
    - core: The core object containing necessary parameters and data.
    - alms: The input alm array.

    Returns:
    - alm_ng: The calculated almng array.
    """

    tr_ells = core.cosmo.transfer["ells"]
    tr_k = core.cosmo.transfer["k"]
    tr_ell_k = core.cosmo.transfer["tr_ell_k"]

    # just trim the ells so we do not go beyond what ells we have for the transfer functions
    ells = core.ells[2:]

    # logger.info("Using ells %s, %s", ells.shape, ells)

    # the following few blocks of code calculate eq 14 and 15 from Smith and Zal.
    # the radian_func does the f_ell^X(r) = (2/pi) int k^2 dk f(k) transfer^X_ell(k) j_ell(k r),
    # where f(k) is an arbitrary function of wavenumber k.

    # A is slightly different due to the KSW code, see the notes in KSW's cosmo.py::add_prim_reduced_bispectrum
    A = (3 / 5) ** 2 * 2 * np.pi**2 * core.cosmo_params["As"]
    delta_phi = (tr_k) ** ((core.cosmo_params["ns"] - 4))

    f_k = np.full((len(tr_k), 2), 5 / 3, dtype=core.r_dtype)
    # f_k[:, 0] *= 1            # f_k for alpha
    f_k[:, 1] *= A * delta_phi  # f_k for beta

    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interpolator(alpha_ell, tr_ells)(ells)

    beta_ell = rad[..., 1]
    beta_l = interpolator(beta_ell, tr_ells)(ells)
    bl_div_cl = beta_l / core.c_ells[ells, : core.npol]

    logger.info("Starting Alm_ng generation")

    drs = np.diff(core.radii)
    alm_ng = np.zeros_like(alms)

    # This uses joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    # I also use return_as generator which allows us to consume the results as they are generated (in order),
    # this helps prevent memory issues
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for Alm_ng generation: {temp_folder}")

    parallel = Parallel(n_jobs=-1, return_as="generator", temp_folder=temp_folder)
    for sim, pol in tqdm(core.sim_pol, total=core.sim_pol_len, desc="Alm_ng"):
        generator = parallel(
            delayed(integrand)(
                alms[sim, pol],
                bl_div_cl[r, :, pol],
                alpha_l[r, :, pol],
                core.nside,
                core.lmax,
                core.radii[r],
            )
            for r in range(len(core.radii))
        )

        alm_ng[sim, pol] += np.trapz(list(generator), core.radii, dx=drs, axis=0)

    return alm_ng


def main():
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
    core = Core()

    # check for completed alm runs if we are not forcing alm generation, and fail fast
    # notice we don't exit(1) so we can use this to resume a partially completed job without having to recalculate everything
    if not core.force_gen:
        if os.path.isfile(core.alm_file):
            logger.warning("Found completed alms file, skipping alm generation")
            sys.exit(0)
        elif os.path.isfile(core.alm_file_partial):
            logger.warning("Found partial alm file, skipping alm generation")
            sys.exit(0)

    # Get our gaussian alms, very fast so no need to parallelize
    logger.info("Starting Alm generation")
    alm_l = np.array(
        [core.data.compute_alm_sim(core.lensing) for _ in range(core.nsims)],  # type: ignore
        dtype=core.c_dtype,
    )
    logger.info("alm_l shape %s", alm_l.shape)

    # get the non-gaussian alms, this will take a long time
    alm_ng = generate_alm_ng(core, alm_l)

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, core.fnl_shape)

    # finally combine into the full alms and remove the monopole and dipole terms by setting them to 0
    alms = alm_l + fnls[:, None, :] * alm_ng
    alms = remove_mono_dipole(alms)

    logger.info("Saving data")
    os.makedirs(core.alm_dir, exist_ok=True)

    # we remove the stale nc file if it exists prior to saving the new one
    if os.path.isfile(core.alm_file_partial):
        logger.info("Removing stale alm file: %s", core.alm_file_partial)
        os.remove(core.alm_file_partial)

    sdata = {}
    sdata["alm"] = alms
    sdata["fnl"] = fnls
    save_data(core.alm_file_partial, sdata)

    if core.is_main_job:
        alm_plot_dir = os.path.join(core.plot_dir, "almgen")
        os.makedirs(alm_plot_dir, exist_ok=True)

        i, j = core.rng.integers(core.nsims), core.rng.integers(core.npol)
        logger.debug("Making plots for alm[%d,%d]", i, j)
        filebase = os.path.join(
            alm_plot_dir, f"{core.sjob}_{core.base_name}_alm[{i},{j}].png"
        )

        # plot the complete alms with noise
        plot_cl_alm(
            alms[i, j],
            save_file=filebase,
            plot_camb=True,
            c_ells=core.c_ells[:, j],
            camb_noise=True,
            noise=core.noise_ell[j],
            beam_width=core.beam_width,
        )

    logger.info("Finished %s!", core.sjob)


if __name__ == "__main__":
    sys.exit(main())
