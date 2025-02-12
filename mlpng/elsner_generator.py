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
        cls = core.cov

    print("genering alm with cls", cls.shape, flush=True)
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

    Balm = np.array([hp.almxfl(alm[p], bl_div_cl[..., p]) for p in range(alm.shape[0])])
    B = hp.alm2map(Balm, nside, pol=False)
    inner = np.array(hp.map2alm(B**2, lmax, use_pixel_weights=True, pol=False), ndmin=2)
    solution = [hp.almxfl(inner[p], alpha_l[..., p]) for p in range(alpha_l.shape[-1])]
    return np.array(solution)


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

    pol_idxs = core.pol_idxs()
    logger.debug("Using pol_idxs: %s", pol_idxs)

    # transfer = compute_transfer(core, core.lmax, verbose=True)
    transfer = core.cosmo.transfer

    # get the transfer functions with tr_ell_k being \Delta_\ell(k)
    tr_ells = transfer["ells"]
    tr_k = transfer["k"]
    tr_ell_k = transfer["tr_ell_k"]  # this is T, E, PHI
    tr_ell_k = tr_ell_k[..., pol_idxs]

    # camb returns transfers in zeta, need to convert to phi
    tr_ell_k *= 5 / 3

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    # see komatsu 2003 eq 5 and 6
    f_k = np.ones((len(tr_k), 2), dtype=core.r_dtype)

    # we need to multiply the f_k by the delta_phi term, and convert from k^2 to k^-1
    # see creminelii 2006, eq 16
    # for delta_phi = A: (taken from adri's ksw)
    # Planck defines A as <phi_k1 phi_k2> = (2pi)^3 delta(k12) A / k^3.
    # CAMB defines As as <zeta_k2 zeta_k2> = (2pi)^3 delta(k12) 2 * pi^2 As / k^3.
    # So A = (3/5)^2 * 2 * pi^2 As.
    Pk = core.cosmo.camb_params.primordial_power(tr_k, 0)
    delta_phi = 2 * np.pi**2 * Pk * (3 / 5) ** 2
    f_k[:, 1] = delta_phi * tr_k ** (-3)
    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interp1d(
        tr_ells,
        alpha_ell,
        "cubic",
        1,
        bounds_error=False,
        fill_value=0,  # type: ignore
    )(core.ells)

    beta_ell = rad[..., 1]
    beta_l = interp1d(
        tr_ells,
        beta_ell,
        "cubic",
        1,
        bounds_error=False,
        fill_value=0,  # type: ignore
    )(core.ells)

    bl_div_cl = np.zeros_like(beta_l)
    bl_div_cl[:, core.lmin :] = beta_l[:, core.lmin :] * core.icov2.T[core.lmin :]

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
                alms[sim, pol_idxs],
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
    pols = core.pol_idxs()

    c_ell = core.c_ell if not lensed else core.c_ell_lensed
    cov = c_ell + core.n_ell / core.b_ell**2

    ib = np.ones_like(c_ell)
    ib[pols, core.lmin :] = 1 / core.b_ell[pols, core.lmin :]

    ic_ell = np.zeros_like(cov)
    ic_ell[pols, core.lmin :] = 1 / cov[pols, core.lmin :]
    ic_ell = ic_ell[pols]

    fisher = core.estimator.compute_fisher_isotropic(ic_ell, comm=None)

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

    elsner_cl_file = os.path.join(core.dirs["base"], "elsner", "cl_wmap5_bao_sn.dat")
    data = np.loadtxt(elsner_cl_file)

    # want to add the mono and dipole terms to match other code
    data = np.vstack([[0, 0, 0, 0], [1, 0, 0, 0], data])

    ells = data[:, 0]
    c_ell_tt = data[:, 1]
    c_ell_ee = data[:, 2]
    c_ell_te = data[:, 3]
    c_ells = np.array([c_ell_tt, c_ell_ee, np.zeros_like(c_ell_tt), c_ell_te])

    # scale = ells * (ells + 1) / (2 * np.pi)
    # c_ells *= scale
    c_ells *= (core.cosmo.camb_params.TCMB * 1e6) ** 2

    # update the core to use the new cls, hacky
    pols = core.pol_idxs()
    core.c_ell = c_ells[:, : core.nell]

    core.cov = cov = core.b_ell**2 * core.c_ell + core.n_ell
    inoise = np.full(core.n_ell.shape, 1e-16)
    inoise[pols, core.lmin :] = 1 / core.n_ell[pols, core.lmin :]

    icov = np.zeros_like(cov)
    icov[pols, core.lmin :] = 1 / cov[pols, core.lmin :]
    # icov = get_itotcov_ell(icov, inoise, core.b_ell)
    # icov = icov[pols, pols]
    core.icov = np.array(icov[:, : core.nell])

    alm_l = generate_alm(core)
    logger.debug("alm_l shape: %s, dtype: %s", alm_l.shape, alm_l.dtype)

    # get the non-gaussian alms, this will take a long time
    logger.debug("Starting non-gaussian alm generation")
    alm_nl = generate_alm_nl(core, alm_l)
    logger.debug("alm_nl shape: %s, dtype: %s", alm_nl.shape, alm_nl.dtype)

    # save the data for jorik and daan
    data = {}
    data["alm_l"] = alm_l[:, pols].astype(core.c_dtype)
    logger.debug(
        "saving jorik alm_l shape: %s, dtype: %s",
        alm_l[:, pols].shape,
        alm_l[:, pols].dtype,
    )
    data["alm_nl"] = alm_nl.astype(core.c_dtype)
    logger.debug("saving jorik alm_nl shape: %s, dtype: %s", alm_nl.shape, alm_nl.dtype)
    tstr = f"_{core.slurm.array_index}" if core.slurm.array_index > 0 else ""
    file = os.path.join(core.dirs["data"], f"alm-{core.name}{tstr}.hdf5")
    save_data(file, data)
    print("done saving for jorik")

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, (core.nsims, core.ndups, 1, 1))

    # Add the duplicate dimension to the alms
    alm_l = alm_l[:, None, core.pol_idxs()]
    alm_nl = alm_nl[:, None]

    alms = alm_l + fnls * alm_nl
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
            make_alm_plots(core, alms=alm_lensed[:, 0, core.pol_idxs()], lensed=True)

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
        sdata["alm_lensed"] = alm_lensed[:, :, core.pol_idxs()].astype(core.c_dtype)
        sdata["map_lensed"] = maps_lensed[:, :, core.pol_idxs()].astype(core.r_dtype)
        sdata["phi_map"] = phi_map.astype(core.r_dtype)

    save_data(core.file, sdata)

    logger.info("Finished %s!", core.slurm.job)


if __name__ == "__main__":
    sys.exit(main())
