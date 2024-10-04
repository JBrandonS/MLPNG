import logging
import os
import sys

import camb
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline, interp1d
from tqdm.auto import trange

import matplotlib.pyplot as plt

import lenspyx
from lenspyx import utils_hp
from pixell import curvedsky, enmap, reproject

from ksw import Cosmology
from ksw.radial_functional import radial_func

from . import Core
from .utils import remove_mono_dipole, save_data, setup_logging
from .utils.plots import (
    plot_cl_alm,
    plot_patches,
    plot_mollview,
    plot_elsner_comp,
    pol_str,
)

logger = setup_logging("generator", level=logging.DEBUG)


def interpolator(func, ells_sparse, axis=1, cubic=True, **kwargs):
    """
    Interpolates a function using either cubic spline or interp1d (linear) interpolation.

    Parameters:
        func (array_like): The function values to be interpolated.
        ells_sparse (array_like): The sparse grid points.
        axis (int, optional): The axis along which to interpolate. Default is 1.
        cubic (bool, optional): If True, cubic spline interpolation is used. If False, linear interpolation is used. Default is True.
        kwargs: Additional keyword arguments to be passed to the interpolating function.

    Returns:
        object: An interpolating function.
    """

    if cubic:
        return CubicSpline(ells_sparse, func, axis, **kwargs)
    else:
        return interp1d(ells_sparse, func, axis=axis, **kwargs)


def generate_alm(core, c_ells, nsims=None):
    """
    generates the gaussian alms using the given core.

    Parameters:
        core: The core object containing necessary parameters and data.
        nsims: The number of simulations to generate. If not provided will use the core's nsims.

    Returns:
        sims: The generated alms. in TT, EE, BB, TE order.
    """
    if nsims is None:
        nsims = core.nsims

    sims = [hp.synalm(c_ells, lmax=core.lmax, new=True) for _ in range(nsims)]
    return np.ascontiguousarray(sims)[:, core.pol_idxs()]


def integrand(alm, alpha_l, bl_div_cl, lmax, nside):
    """
    Calculates the Outer integral of Hansen 2010 eq 27. That is:

    int dr r^2 [ alpha_ell(r) ( int d^2 hat{n} Y^*_{ell m}(hat{n}) B(r, hat{n})^2 ) ]

    Parameters:
        alm (ndarray): The input spherical harmonic coefficients.
        alpha_l (ndarray): The alpha_l coefficients.
        bl_div_cl (ndarray): The division of beta_ell function and C_l.

    Returns:
        ndarray: The calculated integrand.
    """
    npols = alm.shape[0]

    Balm = [hp.almxfl(alm[p], bl_div_cl[..., p]) for p in range(npols)]
    B = hp.alm2map(Balm, nside, lmax, pol=False)
    inner = hp.map2alm(B**2, lmax, use_pixel_weights=True, pol=False)
    inner = np.array(inner, ndmin=2)
    return np.array([hp.almxfl(inner[p], alpha_l[:, p]) for p in range(npols)])


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


def generate_alm_ng(core, cosmo, alms, c_ells):
    """
    This function calculates the non-gaussian alms using the given core and the gaussian alms.

    Parameters:
        core: The core object containing necessary parameters and data.
        alms: The input alm array.

    Returns:
        alm_ng: The calculated almng array.
    """

    # get the transfer functions with tr_ell_k being \Delta_\ell(k)
    tr_ells = cosmo.transfer["ells"]
    tr_k = cosmo.transfer["k"]
    tr_ell_k = cosmo.transfer["tr_ell_k"][..., core.pol_idxs()]  # this is T, E, PHI

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    # see komatsu 2003 eq 5 and 6
    f_k = np.ones((len(tr_k), 2), dtype=core.r_dtype)

    # for beta, get the Pk from camb and convert from the dimensionless Pk from camb to the dimensionful
    Pk = cosmo.camb_params.primordial_power(tr_k, 0)
    # we do not need the tr_k**(ns - 1) term as it is accounted for in the Pk
    f_k[:, 1] = 2 * np.pi**2 * tr_k ** (-3) * Pk

    # this computes \frac{2}{\pi} \int_0^\infty dk k^2 f_k tr_ell_k j_\ell(k r)
    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interpolator(alpha_ell, tr_ells)(core.ells)
    alpha_l[:, : core.lmin] = 0.0  # remove the monopole and dipole terms

    beta_ell = rad[..., 1]
    beta_l = interpolator(beta_ell, tr_ells)(core.ells)

    # next 3 lines prevent a division by zero due to the monopole and dipole terms being 0
    bl_div_cl = np.zeros_like(beta_l)
    bl_div_cl[:, core.lmin :] = (
        beta_l[:, core.lmin :] / c_ells.T[None, core.lmin :, : core.npols]
    )

    # ensure all arrays are contiguous, they wont be since we are using the interpolator
    alpha_l = np.ascontiguousarray(alpha_l)
    bl_div_cl = np.ascontiguousarray(bl_div_cl)

    # This uses joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for Alm_ng generation: {temp_folder}")
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

def make_alm_plots(core, alm_l, alm_ng, alms, c_ells):
    logger.debug("Generating alm plots")
    sim = core.rng.integers(core.nsims)  # get random sim idx

    # plot a few comparison with different functions to get views
    idx = core.rng.integers(1, 1001)
    plot_elsner_comp(
        alm_l[sim],
        alm_ng[sim],
        index=idx,
        save_file=core.get_plot_file("ecomp"),
        plot_func=plt.plot,
    )
    plot_elsner_comp(
        alm_l[sim],
        alm_ng[sim],
        index=idx,
        save_file=core.get_plot_file("ecomp_log"),
        plot_func=plt.loglog,
    )
    plot_elsner_comp(
        alm_l[sim],
        alm_ng[sim],
        index=idx,
        save_file=core.get_plot_file("ecomp_semilogy"),
        plot_func=plt.semilogy,
    )

    # lets plot the alms with camb for testing that side of things
    cl_settings = {
        "plot_camb": True,
        "camb_cls": c_ells,
        "plot_noise": False,
        "plot_full_camb": True,
        "camb_noise": core.noise_ell,
        "camb_beam": core.beam_ell,
    }
    # full alms
    plot_cl_alm(
        alms[sim],
        save_file=core.get_plot_file(f"{sim}_alm"),
        ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}",
        **cl_settings,
    )
    plot_cl_alm(
        alm_l[sim],
        save_file=core.get_plot_file(f"{sim}_alm_l"),
        ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}^{L}$",
        **cl_settings,
    )
    plot_cl_alm(
        alm_ng[sim],
        save_file=core.get_plot_file(f"{sim}_alm_ng"),
        ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}^{NG}$",
        **cl_settings,
    )

def patch_and_lens(core, cosmo, alms):
    logger.debug("Generating patch geometry")
    ps_rad = np.deg2rad(core.patch_side_deg)
    res = ps_rad / core.nside

    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    # need to add a B mode dim if we are using E modes, this is used for some lensing
    full_pol = core.npols + (1 if core.use_e else 0)
    fs_shape = (full_pol,) + fs_shape
    fs_map = enmap.zeros(fs_shape, fs_wcs)

    shapes = []
    wcss = []
    for counter in range(core.npatches // 2):
        # [[dec_min,ra_min],[dec_max,ra_max]]
        top = [[0, ps_rad * counter], [ps_rad, ps_rad * (counter + 1)]]
        gs, w = enmap.geometry(pos=top, res=res, proj="car")
        shapes.append(gs)
        wcss.append(w)

        bottom = [[-ps_rad, ps_rad * counter], [0, ps_rad * (counter + 1)]]
        gs, w = enmap.geometry(pos=bottom, res=res, proj="car")
        shapes.append(gs)
        wcss.append(w)

    patches = None
    alm_lensed = None

    # make a copy of the alms for the lensing since they will modify them
    maps = alms.copy()
    if not core.use_t:
        maps = np.concatenate((np.zeros((core.nsims, 1, core.nelem)), maps), axis=1)
    if core.use_e:
        # need to add a zero for the spin-2 component
        maps = np.concatenate((maps, np.zeros((core.nsims, 1, core.nelem))), axis=1)

    # now lets do the actual lensing and patching
    if core.lensing:
        logger.debug("Getting lensing cl_phi and data")
        # PP PT PE
        cl_phi = cosmo._camb_data.get_lens_potential_cls(core.lmax, "muK", True)
        plm = utils_hp.synalm(cl_phi[:, 0], core.lmax, mmax=None)

        # transform the lensing potential into spin-1 deflection field
        fl = np.sqrt(np.arange(core.nell) * np.arange(1, core.nell + 1))
        dlm = utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

        geom_info = ("healpix", {"nside": core.nside})
        geom = lenspyx.get_geom(geom_info)

        patches = np.zeros(
            (core.nsims, core.npatches, full_pol, core.nside, core.nside),
            dtype=core.r_dtype,
        )
        alm_lensed = np.zeros((core.nsims, full_pol, core.nelem), dtype=core.c_dtype)

        # actually lens the alms and cut the patches
        for sim in trange(core.nsims, desc="Lensing and Patching", total=core.nsims):
            lenmap = lenspyx.alm2lenmap(
                maps[sim], dlm, geometry=geom_info, nthreads=core.n_cpus
            )
            lenmap = np.ascontiguousarray(lenmap)  # convert from tuple to array

            if core.use_t:
                alm_lensed[sim, 0] = geom.map2alm(
                    lenmap[0].copy(), core.lmax, core.lmax, nthreads=core.n_cpus
                )

            if core.use_e:
                idx = 1 if core.use_t else 0
                alm_lensed[sim, idx:] = geom.map2alm_spin(
                    lenmap[1:].copy(), 2, core.lmax, core.lmax, nthreads=core.n_cpus
                )

            # cut the patches
            pixell_map = reproject.healpix2map(
                lenmap[core.pol_idxs(keep_b=True)],
                fs_shape,
                fs_wcs,
                core.lmax,
                spin=np.array([0, 2])[core.pol_idxs()],
            )
            for i in range(core.npatches):
                patches[sim, i] = pixell_map.project(shapes[i], wcss[i])

            if core.plot and core.slurm.is_main:
                sim = core.rng.integers(core.nsims)
                map = lenspyx.alm2lenmap(
                    alms[sim], dlm, geom_info, nthreads=core.n_cpus
                )
                plot_mollview(
                    map,
                    f"Lensed view for {sim}",
                    save_file=core.get_plot_file(f"{sim}-moll"),
                )
    else:
        logger.debug("Getting patches")
        patches = np.zeros(
            (core.nsims, core.npatches, core.npols, core.nside, core.nside),
            dtype=core.r_dtype,
        )

        for sim in trange(core.nsims, desc="Patching", total=core.nsims):
            car_map = curvedsky.alm2map(
                maps[sim],
                fs_map,
                spin=np.array([0, 0])[core.pol_idxs()],
                copy=True,
                nthread=core.n_cpus,
            )

            for i in range(core.npatches):
                patches[sim, i] = car_map.project(shapes[i], wcss[i])[: core.npols]

        # lets make a few plots of patches and mollview
        if core.plot and core.slurm.is_main:
            sim = core.rng.integers(core.nsims)

            map = curvedsky.alm2map_healpix(
                alms[sim],
                nside=core.nside,
                spin=np.array([0, 0])[core.pol_idxs()],
                copy=True,
                nthread=core.n_cpus,
            )
            plot_mollview(
                map,
                f"Unlensed view for {sim}",
                save_file=core.get_plot_file(f"{sim}-moll"),
            )

    # make some plots here
    if core.plot and core.slurm.is_main:
        sim = core.rng.integers(core.nsims)
        for pol in range(core.npols):
            pstr = pol_str(pol)
            plot_patches(
                patches[sim, :, pol],
                title=f"Patches for sim: {sim}, pol: {pstr}",
                save_file=core.get_plot_file(f"{sim}{pstr}-patch.png"),
            )

    return patches, alm_lensed


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

    # setup our cosmology and compute the c_ells
    # this is not pulled into the core due to the nvidia server not liking KSW
    cosmo = Cosmology(camb.set_params(**core.cosmo_params, verbose=False))
    cosmo.compute_transfer(core.lmax)
    cosmo.compute_c_ell()

    if core.lensing:
        c_ells = cosmo.c_ell["lensed_scalar"]["c_ell"].T
    else:
        c_ells = cosmo.c_ell["unlensed_scalar"]["c_ell"].T

    alm_l = generate_alm(core, c_ells)

    # get the non-gaussian alms, this will take a long time
    logger.debug("Starting non-gaussian Alm generation")
    alm_ng = generate_alm_ng(core, cosmo, alm_l, c_ells)

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, (core.nsims,))

    # finally combine into the full alms and remove the monopole and dipole terms
    alms = alm_l + fnls[:, None, None] * alm_ng
    alms = remove_mono_dipole(alms)  # dont think this is needed, but safety
    logger.info("Completed Alm generation!")

    if core.plot and core.slurm.is_main:
        # lets make a few plots for the alms
        make_alm_plots(core, alm_l, alm_ng, alms, c_ells)

    logger.info("Starting lensing and patch generation")
    patches, alm_lensed = patch_and_lens(core, cosmo, alms)

    logger.info("Saving data")
    sdata = {}
    sdata["alm"] = alms
    sdata["fnl"] = fnls
    sdata["patch"] = patches
    if core.lensing:
        sdata["alm_lensed"] = alm_lensed
    save_data(core.file, sdata, remove_if_exists=True)

    logger.info("Finished %s!", core.slurm.job)


if __name__ == "__main__":
    sys.exit(main())
