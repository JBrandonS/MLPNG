"""..."""

import gc
import logging
import os
import sys
from itertools import product
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm.auto import trange

import lenspyx
from lenspyx import utils_hp

from ksw.radial_functional import radial_func

from . import Core, get_itotcov_ell
from .utils import save_data, setup_logging, remove_mono_dipole, make_alm_plots

logger = setup_logging("mlpng.generator", level=logging.DEBUG)


def generate_alm(core, nsims=None, cls=None):
    """
    generates the gaussian alms using the given core.

    Parameters:
        core: The core object containing necessary parameters and data.
        nsims: The number of simulations to generate. If not provided will use the core's nsims.
        cls: The cls to use for the generation. If not provided will use the core's (unlensed) c_ell.

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
    else:
        # TODO FIX
        empty = np.zeros((1, cls.shape[1]))
        cls = np.concatenate((cls[:-1], empty, cls[-1:]), axis=0)
        core.c_ell = cls[:, : core.nell]

    if core.use_e and not core.use_t:
        empty = np.zeros((1, cls.shape[1]))
        cls = np.concatenate((empty, cls, empty), axis=0)

    sims = [hp.synalm(cls, lmax=core.lmax, new=True) for _ in range(nsims)]
    return np.ascontiguousarray(sims)[:, core.pol_idxs()]


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
    Balm = np.array([hp.almxfl(alm[p], bl_div_cl[..., p]) for p in range(npols)])
    if Balm.shape[0] == 2:
        Balm = np.vstack([Balm, np.zeros_like(Balm[0])])

    B = hp.alm2map(Balm, nside, pol=npols == 3)
    inner = hp.map2alm(B**2, lmax, use_pixel_weights=True, pol=npols == 3)
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


def generate_alm_nl(core, alms):
    """
    This function calculates the non-gaussian alms using the given core and the gaussian alms.

    Parameters:
        core: The core object containing necessary parameters and data.
        alms: The input alm array.

    Returns:
        alm_nl: The calculated almng array.
    """

    # get the transfer functions with tr_ell_k being \Delta_\ell(k)
    tr_ells = core.cosmo.transfer["ells"]
    tr_k = core.cosmo.transfer["k"]
    # this is T, E, PHI
    tr_ell_k = core.cosmo.transfer["tr_ell_k"][..., core.pol_idxs()]

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    # see komatsu 2003 eq 5 and 6
    f_k = np.ones((len(tr_k), 2), dtype=core.r_dtype)
    Pk = core.cosmo.camb_params.primordial_power(tr_k, 0)
    f_k[:, 1] = 2 * np.pi**2 * Pk / (tr_k) ** (3)

    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interp1d(
        tr_ells,
        alpha_ell,
        "cubic",
        1,
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )(core.ells)

    beta_ell = rad[..., 1]
    beta_l = interp1d(
        tr_ells,
        beta_ell,
        "cubic",
        1,
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
    )(core.ells)

    # komatsu, first year, 3 and 4
    # bl = core.b_ell.T[None, :, core.pol_idxs(pretrimmed=True)]
    # alpha_l *= bl
    # beta_l *= bl

    s_ell = core.b_ell**2 * core.c_ell + core.n_ell
    # s_ell = core.c_ell + core.n_ell / core.b_ell**2
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
    logger.debug("Using temp folder for alm_nl generation: %s", temp_folder)
    parallel = Parallel(core.n_cpus, return_as="generator", temp_folder=temp_folder)
    alm_nl = np.zeros_like(alms)

    logger.debug("Starting...")
    for sim in trange(core.nsims, desc="alm_nl"):
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
        alm_nl[sim] = trap_generator(generator, core.radii)

    return alm_nl


def lens_alms(core, alms_to_lens):
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

    phi_map = hp.alm2map(plm, nside=core.nside, pol=False)

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
            alm_lensed[sim, dup, 1:] = geom.map2alm_spin(
                lenmap[1:].copy(), 2, core.lmax, core.lmax, nthreads=core.n_cpus
            )
    return alm_lensed, phi_map


def get_fisher_iso(core, lensed=False):
    core.init_estimator()
    pols = core.pol_idxs(keep_b=False, keep_te=False, pretrimmed=True)
    c_ell = core.c_ell if not lensed else core.c_ell_lensed

    ic_ell = np.zeros_like(c_ell)
    ic_ell[pols, core.lmin :] = 1 / c_ell[pols, core.lmin :]

    inoise = np.full(core.n_ell.shape, 1e-16)
    inoise[pols, core.lmin :] = 1 / core.n_ell[pols, core.lmin :]

    icov = get_itotcov_ell(ic_ell, inoise, core.b_ell)
    pols = core.pol_idxs(keep_b=False, keep_te=False)
    icov = icov[pols, pols]

    fisher = core.estimator.compute_fisher_isotropic(icov, comm=None)
    # it improves the saving to return a numpy array
    logger.debug("got %s iso fisher: %s", "lensed" if lensed else "unlensed", fisher)
    return np.array([fisher]).astype(core.r_dtype)


def check_existing_data_file(core):
    """ """
    if os.path.exists(core.file):
        if core.force_gen:
            logger.info("Removing existing data file")
            os.remove(core.file)
        else:
            logger.info("Data file exists, exiting")
            sys.exit(0)

    # we also should check for the full file
    full_file = os.path.join(core.dirs["data"], f"{core.name}.hdf5")
    if os.path.exists(full_file):
        if core.force_gen:
            logger.info("Removing existing full data file")
            os.remove(full_file)
        else:
            logger.info("Found full data file, exiting")
            sys.exit(0)


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
    check_existing_data_file(core)

    logger.info("Starting data generation")
    core.init_estimator()

    if core.nsims > 1000:
        raise ValueError("nsims > 1000 is not supported for elsner simulations")

    els_cl_file = os.path.join(core.dirs["base"], "elsner", "cl_wmap5_bao_sn.dat")
    elsner_cls = np.loadtxt(els_cl_file).T[1:]
    elsner_cls *= (core.cosmo.camb_params.TCMB * 1e6) ** 2
    alm_l = generate_alm(core, cls=elsner_cls)
    logger.debug("alm_l shape: %s, dtype: %s", alm_l.shape, alm_l.dtype)

    # get the non-gaussian alms, this will take a long time
    logger.debug("Starting non-gaussian alm generation")
    alm_nl = generate_alm_nl(core, alm_l)
    logger.debug("alm_nl shape: %s, dtype: %s", alm_nl.shape, alm_nl.dtype)

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, (core.nsims, core.ndups, 1, 1))
    logger.debug("fnls shape: %s, dtype: %s", fnls.shape, fnls.dtype)

    # Add the duplicate dimension to the alms
    alm_l = alm_l[:, None]
    alm_nl = alm_nl[:, None]

    # finally combine into the full alms and remove the monopole and dipole terms
    alms = alm_l + fnls * alm_nl
    alms = remove_mono_dipole(alms, inplace=True)  # safety
    alms = np.ascontiguousarray(alms)
    logger.debug("alms shape: %s, dtype: %s", alms.shape, alms.dtype)

    # we go ahead and save some data so it can be freed, needed for nside > 512 to have good memory usage
    if core.save_alms:
        sdata = {}
        sdata["alm"] = alms.astype(core.c_dtype)
        sdata["alm_l"] = alm_l.astype(core.c_dtype)
        sdata["alm_nl"] = alm_nl.astype(core.c_dtype)
        save_data(core.file, sdata)
    else:
        logger.debug("Not saving alms...")
    logger.info("Completed alm generation!")

    logger.info("Getting maps in nest ordering")
    maps = np.zeros((core.nsims, core.ndups, core.npols, core.npix), dtype=core.r_dtype)
    for sim, dup in product(range(core.nsims), range(core.ndups)):
        rmap = hp.alm2map(alms[sim, dup], nside=core.nside, pol=core.npols == 3)
        maps[sim, dup] = hp.reorder(rmap, r2n=True)
    logger.debug("maps shape: %s, dtype: %s", maps.shape, maps.dtype)

    if core.slurm.is_main and core.plot:
        # lets make a few plots for the alms
        make_alm_plots(core, alm_l[:, 0], alm_nl[:, 0], alms[:, 0])

    # once again save the maps to free up memory
    save_data(core.file, {"map": maps.astype(core.r_dtype)})

    # and actually force the garbage collection, although it should be done automatically
    del maps, alm_l, alm_nl
    gc.collect()

    alm_lensed = maps_lensed = phi_map = None
    if core.lensing:
        logger.info("Starting lensing")
        alm_lensed, phi_map = lens_alms(core, alms)

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

        if core.slurm.is_main and core.plot:
            make_alm_plots(core, alms=alm_lensed[:, 0], lensed=True)

    # note: we save as requested dtype but do calculations in complex128 since healpy needs it for alm2map
    logger.debug("Collecting save data")
    sdata = {}
    if core.slurm.is_main:
        sdata["fisher_iso"] = get_fisher_iso(core, False)
        if core.lensing:
            sdata["fisher_iso_lensed"] = get_fisher_iso(core, True)

    sdata["fnl"] = fnls
    # it could possibly be useful to have a normalized range for fnls so they vary between (-1,1)
    fnl_norm = 2 * ((fnls - core.fnl_min) / (core.fnl_max - core.fnl_min)) - 1
    sdata["fnl_norm"] = fnl_norm.astype(core.r_dtype)

    if core.lensing:
        sdata["alm_lensed"] = alm_lensed.astype(core.c_dtype)
        sdata["map_lensed"] = maps_lensed.astype(core.r_dtype)
        sdata["phi_map"] = phi_map.astype(core.r_dtype)

    save_data(core.file, sdata)

    logger.info("Finished %s!", core.slurm.job)


if __name__ == "__main__":
    sys.exit(main())
