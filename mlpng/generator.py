"""..."""

import logging
import os
import sys
from itertools import product
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm.auto import trange, tqdm

import matplotlib.pyplot as plt

import lenspyx
from lenspyx import utils_hp
from pixell import curvedsky, enmap, reproject

from ksw.radial_functional import radial_func

from . import Core, get_itotcov_ell
from .utils import save_data, setup_logging, remove_mono_dipole
from .utils.plots import (
    plot_cl_alm,
    plot_patches,
    plot_mollview,
    plot_elsner_comp,
    pol_str,
)

logger = setup_logging("mlpng.generator", level=logging.DEBUG)


def interpolator(func, ells_sparse, axis=1, kind="cubic", fill_value="extrapolate"):
    """
    Interpolates a given function over specified sparse points using a specified method.

    Parameters:
    func : array-like
        The function values to interpolate.
    ells_sparse : array-like
        The sparse points at which the function values are known.
    axis : int, optional
        The axis along which to interpolate. Default is 1.
    kind : str, optional
        Specifies the kind of interpolation as a string. Default is 'cubic'.
        Options include 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.
    fill_value : str or float, optional
        Specifies the value to use for points outside the interpolation range.
        Default is 'extrapolate'.
    **kwargs : additional keyword arguments
        Additional arguments to pass to the interpolation function.

    Returns:
    scipy.interpolate.interp1d
        An interpolation function that can be called with new points to obtain interpolated values.
    """
    return interp1d(
        ells_sparse, func, kind, axis, bounds_error=False, fill_value=fill_value
    )


def generate_alm(core, nsims=None, cls=None):
    """
    generates the gaussian alms using the given core.

    Parameters:
        core: The core object containing necessary parameters and data.
        nsims: The number of simulations to generate. If not provided will use the core's nsims.
        cls: The cls to use for the generation. If not provided will use the core's s_ell.

    Returns:
        sims: The generated alms. in TT, EE, BB, TE order.
    """
    if nsims is None:
        nsims = core.nsims

    if cls is None:
        if core.noise:
            cls = core.b_ell**2 * core.c_ell + core.n_ell
        else:
            cls = core.c_ell.copy()

    if core.use_e and not core.use_t:
        empty = np.zeros((1, cls.shape[1]))
        cls = np.concatenate((empty, cls, empty), axis=0)

    sims = [hp.synalm(cls, lmax=core.lmax, new=True) for _ in range(nsims)]
    return np.ascontiguousarray(sims, dtype=core.c_dtype)[:, core.pol_idxs()]


def integrand(alm, alpha_l, bl_div_cl, lmax, nside, upscale=False):
    """
    Calculates the Outer integral of Hansen 2010 eq 27. That is:

    int dr r^2 [ alpha_ell(r) ( int d^2 hat{n} Y^*_{ell m}(hat{n}) B(r, hat{n})^2 ) ]

    Parameters:
        alm (ndarray): The input spherical harmonic coefficients.
        alpha_l (ndarray): The alpha_l coefficients.
        bl_div_cl (ndarray): The division of beta_ell function and C_l.
        lmax (int): The maximum l value.
        nside (int): The nside value.
        upscale (bool, optional): If True, the nside will be increased by 1 increment to improve calculations.

    Returns:
        ndarray: The calculated integrand.
    """
    if upscale:
        if nside >= 4096:
            raise ValueError(f"Nside {nside} is too large for upscaling")

        # valid nsides for use_pixel_weights
        nsides = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
        nside = nsides[nsides > nside][0]

    npols = alm.shape[0]
    Balm = [hp.almxfl(alm[p], bl_div_cl[..., p]) for p in range(npols)]
    B = hp.alm2map(Balm, nside, pol=False)
    # TODO, should this be pol=True,
    inner = hp.map2alm(B**2, lmax, use_pixel_weights=True, pol=False)
    inner = np.array(inner, ndmin=2)
    return np.array([hp.almxfl(inner[p], alpha_l[..., p]) for p in range(npols)])


def trap_generator(generator, radii):
    """
    Perform trapezoidal integration using a generator. This will consume the memory as possible to help
    with memory management, this becomes needed for nside >= 512.

    Parameters:
        generator: A generator yielding the function values to integrate.
        radii: The radii values, should match the size of the generator

    Returns:
        The integral computed using Simpson's rule.
    """
    integral = 0.0
    y_prev = radii[0] ** 2 * next(generator)
    for i in range(1, len(radii)):
        y_curr = radii[i] ** 2 * next(generator)
        integral += (radii[i] - radii[i - 1]) * (y_prev + y_curr) / 2.0
        y_prev = y_curr

    return integral


def generate_alm_ng(core, alms):
    """
    This function calculates the non-gaussian alms using the given core and the gaussian alms.

    Parameters:
        core: The core object containing necessary parameters and data.
        alms: The input alm array.

    Returns:
        alm_ng: The calculated almng array.
    """

    # get the transfer functions with tr_ell_k being \Delta_\ell(k)
    tr_ells = core.cosmo.transfer["ells"]
    tr_k = core.cosmo.transfer["k"]
    # this is T, E, PHI
    tr_ell_k = core.cosmo.transfer["tr_ell_k"][
        ..., core.pol_idxs(keep_b=False, keep_te=False)
    ]

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    # see komatsu 2003 eq 5 and 6
    f_k = np.ones((len(tr_k), 2), dtype=core.r_dtype)
    Pk = core.cosmo.camb_params.primordial_power(tr_k, 0)
    f_k[:, 1] = 2 * np.pi**2 * Pk / (tr_k) ** (3)

    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interpolator(alpha_ell, tr_ells)(core.ells)

    beta_ell = rad[..., 1]
    beta_l = interpolator(beta_ell, tr_ells)(core.ells)

    s_ell = core.c_ell + core.n_ell / core.b_ell**2
    s_ell = s_ell[core.pol_idxs(keep_b=False, keep_te=False, pretrimmed=True)]

    icov = np.zeros_like(s_ell)
    icov[:, core.lmin :] = 1 / s_ell[:, core.lmin :]
    bl_div_cl = np.zeros_like(beta_l)
    bl_div_cl[:, core.lmin :] = beta_l[:, core.lmin :] * icov.T[None, core.lmin :]

    # ensure all arrays are c contiguous, they wont be since we are using the interpolator which returns f contiguous
    alpha_l = np.ascontiguousarray(alpha_l)
    bl_div_cl = np.ascontiguousarray(bl_div_cl)

    # This uses joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug("Using temp folder for Alm_ng generation: %s", temp_folder)
    parallel = Parallel(core.n_cpus, return_as="generator", temp_folder=temp_folder)
    alm_ng = np.zeros_like(alms)

    logger.debug("Starting...")
    for sim in trange(core.nsims, desc="Alm_ng"):
        generator = parallel(
            delayed(integrand)(
                alms[sim],
                alpha_l[r],
                bl_div_cl[r],
                core.lmax,
                core.nside,
            )
            for r in range(len(core.radii))
        )

        # here we use our function to calculate the integral
        alm_ng[sim] = trap_generator(generator, core.radii)

    return alm_ng


def make_alm_plots(core, alm_l, alm_ng, alms):
    """
    Generate and save various alm plots for comparison and testing.

    Parameters:
        core (object): Core object containing necessary methods and attributes for plotting.
        alm_l (array): Array of linear alm values.
        alm_ng (array): Array of non-Gaussian alm values.
        alms (array): Array of full alm values.
    """
    logger.debug("Generating alm plots")
    sim = core.rng.integers(core.nsims)  # get random sim idx

    # plot a few comparison with different functions to get views
    idx = core.rng.integers(1, 1001)
    plot_elsner_comp(
        core,
        alm_l[sim],
        alm_ng[sim],
        index=idx,
        save_file=core.get_plot_file("ecomp"),
        plot_func=plt.plot,
        elsner_pols=core.pol_idxs(),
    )
    plot_elsner_comp(
        core,
        alm_l[sim],
        alm_ng[sim],
        index=idx,
        save_file=core.get_plot_file("ecomp_log"),
        plot_func=plt.loglog,
        elsner_pols=core.pol_idxs(),
    )
    plot_elsner_comp(
        core,
        alm_l[sim],
        alm_ng[sim],
        index=idx,
        save_file=core.get_plot_file("ecomp_semilogy"),
        plot_func=plt.semilogy,
        elsner_pols=core.pol_idxs(),
    )

    plot_cl_alm(
        core,
        alms[sim],
        save_file=core.get_plot_file(f"{sim}_alm"),
        ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}",
        plot_camb=True,
        plot_full_camb=True,
    )
    plot_cl_alm(
        core,
        alm_l[sim],
        save_file=core.get_plot_file(f"{sim}_alm_l"),
        ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}^{L}$",
        plot_camb=True,
        plot_full_camb=True,
    )
    plot_cl_alm(
        core,
        alm_ng[sim],
        save_file=core.get_plot_file(f"{sim}_alm_ng"),
        ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}^{NG}$",
    )


def lens_alms(core, alms_to_lens):
    """ """
    full_pol = core.npols + (1 if core.use_e and core.lensing else 0)

    # make a copy of the alms for the lensing since they will modify them
    alm = alms_to_lens.copy()
    if not core.use_t:
        alm = np.concatenate(
            (np.zeros((core.nsims, core.ndups, 1, core.nelem)), alm), axis=2
        )
    if core.use_e:
        # need to add a zero for the spin-2 component
        alm = np.concatenate(
            (alm, np.zeros((core.nsims, core.ndups, 1, core.nelem))), axis=2
        )
    alm_lensed = np.zeros_like(alm)

    # PP PT PE
    cl_phi = core.cosmo._camb_data.get_lens_potential_cls(core.lmax, "muK", True)
    plm = utils_hp.synalm(cl_phi[:, 0], core.lmax, mmax=None)

    # transform the lensing potential into spin-1 deflection field
    fl = np.sqrt(np.arange(core.nell) * np.arange(1, core.nell + 1))
    dlm = utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

    geom_info = ("healpix", {"nside": core.nside})
    geom = lenspyx.get_geom(geom_info)

    for sim, dup in product(range(core.nsims), range(core.ndups)):
        lenmap = lenspyx.alm2lenmap(
            alm[sim, dup], dlm, geometry=geom_info, nthreads=core.n_cpus
        )
        lenmap = np.ascontiguousarray(lenmap)  # convert from tuple to array

        if core.use_t:
            alm_lensed[sim, dup, 0] = geom.map2alm(
                lenmap[0].copy(), core.lmax, core.lmax, nthreads=core.n_cpus
            )

        if core.use_e:
            # print(
            #     "alm_lensed", alm_lensed.shape, "lenmap", lenmap.shape, flush=True
            # )
            # idx = 1 if core.use_t else 0
            alm_lensed[sim, dup, 1:] = geom.map2alm_spin(
                lenmap[1:].copy(), 2, core.lmax, core.lmax, nthreads=core.n_cpus
            )
    return alm_lensed

def get_fisher_iso(core):
    core.init_estimator()
    pols = core.pol_idxs(keep_b=False, keep_te=False, pretrimmed=True)

    ic_ell = np.zeros_like(core.c_ell)
    ic_ell[pols, core.lmin :] = 1 / core.c_ell[pols, core.lmin :]

    inoise = np.full(core.n_ell.shape, 1e-16)
    inoise[pols, core.lmin :] = 1 / core.n_ell[pols, core.lmin :]

    icov = get_itotcov_ell(ic_ell, inoise, core.b_ell)
    pols = core.pol_idxs(keep_b=False, keep_te=False)
    icov = icov[pols, pols]

    fisher = core.estimator.compute_fisher_isotropic(icov, comm=None)
    # it improves the saving to return a numpy array
    return np.array([fisher])


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

    where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions.
    It then will optionally lens the alms and finally it will cut them into patches
    """
    core = Core()

    # check for existing data files, if we are forcing generation, remove them
    # otherwise we exit with 0 so the script can keep going
    if os.path.exists(core.file):
        if core.force_gen:
            logger.info("Removing existing data file")
            os.remove(core.file)
        else:
            logger.info("Data file exists, exiting")
            sys.exit(0)

    logger.info("Starting data generation")

    core.init_estimator()

    alm_l = generate_alm(core)
    logger.debug("alm_l shape: %s, dtype: %s", alm_l.shape, alm_l.dtype)

    # get the non-gaussian alms, this will take a long time
    logger.debug("Starting non-gaussian alm generation")
    alm_ng = generate_alm_ng(core, alm_l)
    logger.debug("alm_ng shape: %s, dtype: %s", alm_ng.shape, alm_ng.dtype)

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, (core.nsims, core.ndups, 1, 1))
    logger.debug("fnls shape: %s, dtype: %s", fnls.shape, fnls.dtype)

    # Add the duplicate dimension to the alms
    alm_l = alm_l[:, None]
    alm_ng = alm_ng[:, None]

    # finally combine into the full alms and remove the monopole and dipole terms
    alms = alm_l + fnls * alm_ng
    alms = remove_mono_dipole(alms, inplace=True)  # safety
    alms = np.ascontiguousarray(alms)
    logger.debug("alms shape: %s, dtype: %s", alms.shape, alms.dtype)

    logger.info("Completed alm generation!")

    logger.info("Getting maps in nest ordering")
    maps = np.zeros((core.nsims, core.ndups, core.npols, core.npix), dtype=core.r_dtype)
    for sim, dup in product(range(core.nsims), range(core.ndups)):
        rmap = hp.alm2map(alms[sim, dup], nside=core.nside, pol=core.npols == 3)
        maps[sim, dup] = hp.reorder(rmap, r2n=True)
    logger.debug("maps shape: %s, dtype: %s", maps.shape, maps.dtype)

    if core.plot and core.slurm.is_main:
        # lets make a few plots for the alms
        make_alm_plots(core, alm_l[:, 0], alm_ng[:, 0], alms[:, 0])

    alm_lensed = None
    maps_lensed = None
    if core.lensing:
        logger.info("Starting lensing")
        alm_lensed = lens_alms(core, alms)

        maps_lensed = np.zeros(
            (core.nsims, core.ndups, 3, core.npix), dtype=core.r_dtype
        )

        logger.debug("Getting lensed maps in nest ordering")
        for sim, dup in product(range(core.nsims), range(core.ndups)):
            rmap = hp.alm2map(
                alm_lensed[sim, dup], nside=core.nside, pol=core.npols == 3
            )
            maps_lensed[sim, dup] = hp.reorder(rmap, r2n=True)
        logger.debug(
            "lensed maps shape: %s, dtype: %s", maps_lensed.shape, maps_lensed.dtype
        )

    # note: we save as requested dtype but do calculations in complex128 since healpy needs it for alm2map
    logger.debug("Collecting save data")
    sdata = {}
    if core.save_alms:
        sdata["alm"] = alms.astype(core.c_dtype)
        sdata["alm_l"] = alm_l.astype(core.c_dtype)
        sdata["alm_ng"] = alm_ng.astype(core.c_dtype)
        if core.lensing:
            sdata["alm_lensed"] = alm_lensed.astype(core.c_dtype)
    else:
        logger.debug("Not saving alms...")

    if core.slurm.is_main:
        # only want one copy of this so only do it on the main thread
        logger.debug("Getting fisher iso")
        sdata["fisher_iso"] = get_fisher_iso(core)

    sdata["fnl"] = fnls
    # it could possibly be useful to have a normalized range for fnls so they vary between (-1,1)
    fnl_norm = 2 * ((fnls - core.fnl_min) / (core.fnl_max - core.fnl_min)) - 1
    sdata["fnl_norm"] = fnl_norm.astype(core.r_dtype)

    sdata["map"] = maps.astype(core.r_dtype)
    if core.lensing:
        sdata["map_lensed"] = maps_lensed.astype(core.r_dtype)

    logger.info("Saving data to %s", core.file)
    save_data(core.file, sdata, remove_if_exists=True)

    logger.info("Finished %s!", core.slurm.job)


if __name__ == "__main__":
    sys.exit(main())
