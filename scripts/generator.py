import logging
import os
import sys

import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline, interp1d
from tqdm.auto import tqdm, trange

import lenspyx
from pixell import curvedsky, enmap, reproject

from ksw.radial_functional import radial_func

from . import Core
from .utils import remove_mono_dipole, save_data, setup_logging
from .utils.plots import (
    plot_cl_alm,
    plot_patches,
    # plot_mollview,
    plot_elsner_comp,
    pol_str,
)

logger = setup_logging("generator", level=logging.DEBUG)


def interpolator(func, ells_sparse, axis=1, cubic=True):
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


def integrand(alm, bl_div_cl, alpha_l, nside, lmax, use_e, radii):
    """
    Calculates the Outer integral of Hansen 2010 eq 27. That is:

    int dr r^2 [ alpha_ell(r) ( int d^2 hat{n} Y^*_{ell m}(hat{n}) B(r, hat{n})^2 ) ]

    Parameters:
        alm (ndarray): The input spherical harmonic coefficients.
        bl_div_cl (ndarray): The division of beta_ell function and C_l.
        alpha_l (ndarray): The alpha_l coefficients.
        nside (int): The HEALPix nside parameter.
        lmax (int): The maximum multipole moment.
        radii (float): the radii at which we are evaluating

    Returns:
        ndarray: The calculated integrand.

    """

    npols = alm.shape[0]
    Balm = [hp.almxfl(alm[pol], bl_div_cl[..., pol]) for pol in range(npols)]

    # We need to shape the Balms correctly
    if use_e:
        if npols == 1:  # will need the T modes
            Balm.insert(0, np.zeros_like(Balm[0]))
        Balm.append(np.zeros_like(Balm[0]))  # add zeros for the b_modes
    elif npols == 1:
        Balm = Balm[0]
    Balm = np.ascontiguousarray(Balm)

    B = hp.alm2map(Balm, nside, lmax)
    inner = hp.map2alm(B**2, lmax, use_pixel_weights=True)
    inner = np.ascontiguousarray(np.atleast_2d(inner))

    results = [hp.almxfl(inner[pol], alpha_l[:, pol]) for pol in range(npols)]
    return radii**2 * np.array(results)


def generate_alm(core, nsims=None):
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

    sims = [hp.synalm(core.c_ells, lmax=core.lmax, new=True) for _ in range(nsims)]
    return np.array(sims)[:, core.pol_idxs()]


def trap_generator(generator, x):
    """
    Perform trapezoidal integration using a generator. This will consume the memory as possible to help
    with memory management, this becomes needed for nside >= 512.

    Parameters:
        generator: A generator yielding the function values to integrate.
        x: The x values corresponding to the function values.

    Returns:
        The integral computed using Simpson's rule.
    """
    integral = 0.0
    y_prev = next(generator)
    for i in range(1, len(x)):
        y_curr = next(generator)
        integral += (x[i] - x[i - 1]) * (y_prev + y_curr) / 2.0
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

    ells = core.ells
    radii = core.radii
    lmin = 2

    # get the transfer functions with tr_ell_k being \Delta_\ell(k)
    tr_ells = core.cosmo.transfer["ells"]
    tr_k = core.cosmo.transfer["k"]
    tr_ell_k = core.cosmo.transfer["tr_ell_k"][..., :2]  # this is T, E, PHI

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    # see komatsu 2003 eq 5 and 6
    f_k = np.ones((len(tr_k), 2), dtype=core.r_dtype)

    # for beta, get the Pk from camb and convert from the dimensionless Pk from camb to the dimensionful
    Pk = core.cosmo.camb_params.primordial_power(tr_k, 0)
    f_k[:, 1] = 2 * np.pi**2 * tr_k ** (-3) * Pk

    # this computes \frac{2}{\pi} \int_0^\infty dk k^2 f_k tr_ell_k j_\ell(k r)
    rad = radial_func(f_k, tr_ell_k, tr_k, radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interpolator(alpha_ell, tr_ells)(ells)

    beta_ell = rad[..., 1]
    beta_l = interpolator(beta_ell, tr_ells)(ells)

    # next 3 lines prevent a division by zero due to the monopole and dipole terms being 0
    bl_div_cl = np.zeros_like(beta_l)
    bl_div_cl[:, lmin:] = beta_l[:, lmin:] / core.c_ells.T[None, lmin:, : core.npols]

    # ensure all arrays are contiguous
    alpha_l = np.ascontiguousarray(alpha_l)
    bl_div_cl = np.ascontiguousarray(bl_div_cl)

    # This uses joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for Alm_ng generation: {temp_folder}")
    parallel = Parallel(core.n_cpus, return_as="generator", temp_folder=temp_folder)
    # parallel = Parallel(1, return_as="generator", temp_folder=temp_folder)
    alm_ng = np.zeros_like(alms)

    logger.debug("Starting...")
    for sim in tqdm(range(alm_ng.shape[0]), total=alm_ng.shape[0], desc="Alm_ng"):
        generator = parallel(
            delayed(integrand)(
                alms[sim],
                bl_div_cl[r],
                alpha_l[r],
                core.nside,
                core.lmax,
                core.use_e,
                radii[r],
            )
            for r in range(len(radii))
        )

        # here we use our function to calculate the integral
        alm_ng[sim] = trap_generator(generator, x=radii)

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

    where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions.
    It then will optionally lens the alms and finally it will cut them into patches
    """
    core = Core()

    # check for existing data files, if we are forcing generation, remove them
    # otherwise we exit with 0 so the script can keep going
    if os.path.exists(core.file_partial):
        if core.force_gen:
            logger.info("Removing existing partial file")
            os.remove(core.file_partial)
        else:
            logger.info("Partial data file exists, exiting")
            sys.exit(0)
    if os.path.exists(core.file_complete):
        if core.force_gen:
            logger.info("Removing existing partial file")
            os.remove(core.file_complete)
        else:
            logger.info("Data file exists, exiting")
            sys.exit(0)

    # Get our gaussian alms, very fast so no need to parallelize
    logger.info("Starting data generation")
    alm_l = generate_alm(core)

    # get the non-gaussian alms, this will take a long time
    logger.debug("Starting non-gaussian Alm generation")
    alm_ng = generate_alm_ng(core, alm_l)

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, core.fnl_shape)

    # finally combine into the full alms and remove the monopole and dipole terms
    alms = alm_l + fnls * alm_ng
    alms = remove_mono_dipole(alms)

    # lets make a few plots for the alms
    if core.is_main_job:
        sim = core.rng.integers(core.nsims)  # get random sim idx
        plot_dir = os.path.join(core.plot_dir, "generator")
        os.makedirs(plot_dir, exist_ok=True)

        # here we compaire to the exact elsner sims from elsner
        elsner_file = os.path.join(plot_dir, f"{core.base_name}_{core.sjob}_elsner.png")
        plot_elsner_comp(
            alm_l[sim],
            alm_ng[sim],
            elsner_idx=core.rng.integers(1, 1001),
            save_file=elsner_file,
        )

        for pol in range(core.npols):
            pstr = pol_str(pol)

            # lets plot the alms with camb for testing that side of things
            filebase = os.path.join(
                plot_dir,
                f"{core.base_name}_{core.sjob}_alm[{sim},{pstr}].png",
            )
            plot_cl_alm(
                alms[sim, pol],
                save_file=filebase,
                plot_camb=True,
                camb_cls=core.c_ells[pol],
                plot_noise=False,
                plot_full_camb=True,
                camb_noise=core.noise_ell[pol],
                camb_beam=core.beam_ell[pol],
                ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}" + f"^{pstr}$",
            )

    logger.info("Completed Alm generation!")

    logger.info("Starting lensing and patch generation")
    logger.debug("Generating patch geometry")
    ps_rad = np.deg2rad(core.patch_side_deg)
    res = ps_rad / core.nside

    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    # need to add a B mode dim if we are using E modes, this is used for some lensing
    full_pol = core.npols + (1 if core.use_e else 0)
    fs_shape = (full_pol,) + fs_shape
    fs_map = enmap.zeros(fs_shape, fs_wcs)

    patch_shapes = []
    patch_wcss = []
    for counter in range(core.npatches // 2):
        # [[dec_min,ra_min],[dec_max,ra_max]]
        top = [[0, ps_rad * counter], [ps_rad, ps_rad * (counter + 1)]]
        gs, w = enmap.geometry(pos=top, res=res, proj="car")
        patch_shapes.append(gs)
        patch_wcss.append(w)

        bottom = [[-ps_rad, ps_rad * counter], [0, ps_rad * (counter + 1)]]
        gs, w = enmap.geometry(pos=bottom, res=res, proj="car")
        patch_shapes.append(gs)
        patch_wcss.append(w)

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
        cl_phi = core.cosmo._camb_data.get_lens_potential_cls(  # type: ignore
            core.lmax, CMB_unit="muK", raw_cl=True
        )
        plm = lenspyx.utils_hp.synalm(cl_phi[:, 0], core.lmax, mmax=None)

        # transform the lensing potential into spin-1 deflection field
        fl = np.sqrt(np.arange(core.nell) * np.arange(1, core.nell + 1))
        dlm = lenspyx.utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

        geom_info = ("healpix", {"nside": core.nside})
        geom = lenspyx.get_geom(geom_info)

        logger.debug("Lensing alms and getting patches")
        # create our data arrays, using empty here for speed and notice forcing npols to 3 for B mode
        patches = np.zeros(
            (core.nsims, core.npatches, full_pol, core.nside, core.nside),
            dtype=core.r_dtype,
        )
        alm_lensed = np.zeros((core.nsims, full_pol, core.nelem), dtype=core.c_dtype)

        # actually lens the alms and cut the patches
        for sim in trange(core.nsims, desc="Lensing and Patching", total=core.nsims):
            lenmap = lenspyx.alm2lenmap(
                maps[sim].copy(), dlm, geometry=geom_info, nthreads=core.n_cpus
            )
            lenmap = np.array(lenmap)  # convert from tuple to array

            if core.use_t:
                alm_lensed[sim, 0] = geom.map2alm(
                    lenmap[0].copy(), core.lmax, core.lmax, nthreads=core.n_cpus
                )

            if core.use_e:
                idx = 1 if core.use_t else 0
                alm_lensed[sim, idx:] = geom.map2alm_spin(
                    lenmap[1:].copy(), 2, core.lmax, core.lmax, nthreads=core.n_cpus
                )

            # we need to build the spin matrix paramters, T = 0, E = 2
            spin = []
            if core.use_t:
                spin.append(0)
            if core.use_e:
                spin.append(2)

            # cut the patches
            pixell_map = reproject.healpix2map(
                lenmap[core.pol_idxs(keep_b=True)],
                fs_shape,
                fs_wcs,
                core.lmax,
                spin=spin,
            )
            for i in range(core.npatches):
                patches[sim, i] = pixell_map.project(patch_shapes[i], patch_wcss[i])
    else:
        logger.debug("Getting patches")
        patches = np.zeros(
            (core.nsims, core.npatches, core.npols, core.nside, core.nside),
            dtype=core.r_dtype,
        )

        # set spin transforms to 0 for the unlensed case, not sure if this is right
        spin = np.atleast_1d(np.zeros(maps.shape[1], dtype=int))

        for sim in trange(core.nsims, desc="Patching", total=core.nsims):
            car_map = curvedsky.alm2map(
                maps[sim], fs_map, spin=spin, copy=True, nthread=core.n_cpus
            )

            for i in range(core.npatches):
                patches[sim, i] = car_map.project(patch_shapes[i], patch_wcss[i])[
                    : core.npols
                ]

    # lets make a few plots of patches and mollview
    if core.is_main_job:
        sim = core.rng.integers(core.nsims)
        plot_dir = os.path.join(core.plot_dir, "generator")
        os.makedirs(plot_dir, exist_ok=True)

        # make the full sky mollview plots
        # plot_file = os.path.join(
        #     plot_dir,
        #     f"{core.base_name}_{core.sjob}_mollview[{sim}].png",
        # )
        # if core.lensing:
        #     map = lenspyx.alm2lenmap(alms[sim].copy(), dlm, geom_info, nthreads=core.n_cpus)
        #     plot_mollview(map, f"Lensed view for {sim}", save_file=plot_file)
        # else:
        #     map = curvedsky.alm2map_healpix(
        #         alms[sim], nside=core.nside, copy=True, nthread=core.n_cpus
        #     )
        #     plot_mollview(map, f"Unlensed view for {sim}", save_file=plot_file)

        for pol in range(core.npols):
            pstr = pol_str(pol)

            # make some platch plots for the sky cuts
            patch_file = os.path.join(
                plot_dir,
                f"{core.base_name}_{core.sjob}_patch[{sim},{pstr}].png",
            )
            plot_patches(
                patches[sim, :, pol],
                title=f"Patches for sim: {sim}, pol: {pstr}",
                save_file=patch_file,
            )

    logger.info("Saving data")
    sdata = {}
    sdata["alm"] = alms
    sdata["fnl"] = fnls
    sdata["patch"] = patches
    if core.lensing:
        sdata["alm_lensed"] = alm_lensed
    save_data(core.file_partial, sdata, remove_if_exists=True)
    logger.info("Finished %s!", core.sjob)


if __name__ == "__main__":
    sys.exit(main())
