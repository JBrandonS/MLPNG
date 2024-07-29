import logging
import os
import sys

import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import simpson
from tqdm.auto import tqdm

import lenspyx
from pixell import curvedsky, enmap, reproject

from ksw.radial_functional import radial_func

from . import Core
from .utils import remove_mono_dipole, save_data, setup_logging
from .utils.plots import plot_cl_alm, plot_patches

logger = setup_logging(__name__, level=logging.DEBUG)


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


def integrand(alm, bl_div_cl, alpha_l, nside, lmax, radii):
    """
    Calculate the integrand for a given set of parameters.

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
    npols = bl_div_cl.shape[-1]
    Balm = [hp.almxfl(alm[pol], bl_div_cl[..., pol]) for pol in range(npols)]
    if npols == 2:
        # add zeros for the b_modes
        Balm.append(np.zeros_like(Balm[0]))
    elif npols == 1:
        Balm = Balm[0]
    Balm = np.ascontiguousarray(Balm)

    B = hp.alm2map(Balm, nside, lmax)
    inner = hp.map2alm(B**2, lmax, use_pixel_weights=True)
    if npols == 1:
        inner = [inner]
    inner = np.ascontiguousarray(inner)

    results = np.array([hp.almxfl(inner[pol], alpha_l[:, pol]) for pol in range(npols)])
    return radii**2 * results


def generate_alm(core, nsims=None):
    """
    generates the gaussian alms using the given core.

    Parameters:
    - core: The core object containing necessary parameters and data.
    - nsims: The number of simulations to generate. Default is None.

    Returns:
    - sims: The generated alms.
    """
    if nsims is None:
        nsims = core.nsims

    totcov = core.c_ells.copy()
    totcov *= core.beam_ell**2
    totcov += core.noise_ell

    # Turn into correct shape
    if not core.use_pols:
        totcov = totcov[0, :]
        totcov = totcov[np.newaxis, :]

    # ensure totcov is contiguous (all in the same block of memory) to make computations faster
    totcov = np.ascontiguousarray(totcov)
    sims = [hp.synalm(totcov, new=True)[: core.npol] for _ in range(nsims)]
    return np.array(sims)


def generate_alm_ng(core, alms):
    """
    This function calculates the non-gaussian alms using the given core and the gaussian alms.

    Parameters:
    - core: The core object containing necessary parameters and data.
    - alms: The input alm array.

    Returns:
    - alm_ng: The calculated almng array.
    """

    logger.debug("Setting up the non-gaussian alms")
    ells = core.ells  # drop the BB mode
    lmin = 2

    tr_ells = core.cosmo.transfer["ells"]
    tr_k = core.cosmo.transfer["k"]
    # transfers are T, E, PHI and not TEB
    tr_ell_k = core.cosmo.transfer["tr_ell_k"][..., : core.npol]

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    f_k = np.ones((len(tr_k), 2), dtype=core.r_dtype)
    pk = core.cosmo.camb_params.primordial_power(tr_k, 0)
    # f_k[:, 0] = 1
    f_k[:, 1] = 2 * np.pi**2 / (tr_k ** (4 - core.cosmo_params["ns"])) * pk * 3 / 5

    # the radian_func does the f_ell^X(r) = (2/pi) int k^2 dk f(k) transfer^X_ell(k) j_ell(k r),
    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interpolator(alpha_ell, tr_ells)(ells)

    beta_ell = rad[..., 1]
    beta_l = interpolator(beta_ell, tr_ells)(ells)

    # divide beta_l by the C_ells to save time later
    # next 3 lines prevent a division by zero dues to the monopole and dipole
    bl_div_cl = np.zeros_like(beta_l)
    c_ells = core.c_ells[: core.npol, lmin:].transpose()
    bl_div_cl[:, lmin:, : core.npol] = beta_l[:, lmin:, : core.npol] / c_ells

    # ensure all arrays are contiguous
    alpha_l = np.ascontiguousarray(alpha_l)
    bl_div_cl = np.ascontiguousarray(bl_div_cl)
    logger.debug("Done")

    # This uses joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for Alm_ng generation: {temp_folder}")
    parallel = Parallel(core.n_cpus, return_as="generator", temp_folder=temp_folder)
    alm_ng = np.empty_like(alms)

    logger.debug("Starting...")
    for sim in tqdm(range(alm_ng.shape[0]), total=alm_ng.shape[0], desc="Alm_ng"):
        generator = parallel(
            delayed(integrand)(
                alms[sim],
                bl_div_cl[r],
                alpha_l[r],
                core.nside,
                core.lmax,
                core.radii[r],
            )
            for r in range(len(core.radii))
        )

        # TODO: Figure out how to do this inline without needing the full generator as required by list
        alm_ng[sim] = simpson(list(generator), x=core.radii, axis=0)

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
    logger.info("Starting Alm generation")
    alm_l = generate_alm(core)
    logger.info("Done!")

    # get the non-gaussian alms, this will take a long time
    logger.info("Starting non-gaussian Alm generation")
    alm_ng = generate_alm_ng(core, alm_l)
    logger.info("Done!")

    # generate the fnls
    fnls = core.rng.uniform(core.fnl_min, core.fnl_max, core.fnl_shape)

    # finally combine into the full alms and remove the monopole and dipole terms
    alms = alm_l + fnls[:, np.newaxis, :] * alm_ng
    alms = remove_mono_dipole(alms)

    logger.info("Completed Alm generation!")
    logger.info("Starting patch generation")

    logger.debug("Generating patch geometry")
    ps_rad = np.deg2rad(core.patch_side_deg)
    res = ps_rad / core.nside

    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    # need to add a B mode dim if we are using E modes, this is used for some lensing
    full_pol = 3 if core.use_pols else core.npol
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
    logger.debug("Done")

    if core.lensing:
        logger.debug("Getting lensing cl_phi and data")
        cl_phi = core.cosmo._camb_data.get_lens_potential_cls(  # type: ignore
            core.lmax, CMB_unit="muK", raw_cl=True
        )
        plm = lenspyx.utils_hp.synalm(cl_phi[:, 0], core.lmax, mmax=None)

        # transform the lensing potential into spin-1 deflection field
        fl = np.sqrt(np.arange(core.lmax + 1) * np.arange(1, core.lmax + 2))
        dlm = lenspyx.utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

        geom_info = ("healpix", {"nside": core.nside})
        geom = lenspyx.get_geom(geom_info)
        logger.debug("Done")

        logger.debug("Lensing alms and getting patches")
        # create our data arrays, using empty here for speed and notice forcing npols to 3 for B mode
        patches = np.empty(
            (core.nsims, core.npatches, full_pol, core.nside, core.nside),
            dtype=core.r_dtype,
        )
        alm_lensed = np.empty((core.nsims, full_pol, core.nelem), dtype=core.c_dtype)

        # actually lens the alms and cut the patches
        for sim in tqdm(
            range(core.nsims), desc="Lensing and Patching", total=core.nsims
        ):
            lens_map = lenspyx.alm2lenmap(
                alms[sim], dlm, geometry=geom_info, nthreads=core.n_cpus
            )
            alm_lensed[sim, 0] = geom.map2alm(
                lens_map[0], core.lmax, core.lmax, nthreads=core.n_cpus
            )
            if core.use_pols:
                alm_lensed[sim, 1:] = geom.map2alm_spin(
                    lens_map[1:], 2, core.lmax, core.lmax, nthreads=core.n_cpus
                )

            # cut the patches
            pixell_map = reproject.healpix2map(lens_map, fs_shape, fs_wcs, core.lmax)
            for i in range(core.npatches):
                patches[sim, i] = pixell_map.project(patch_shapes[i], patch_wcss[i])

        # lets make a plot
        if core.is_main_job:
            import matplotlib.pyplot as plt

            sim = core.rng.integers(core.nsims)
            # Create a figure with subplots
            _, axes = plt.subplots(
                1, core.npol, figsize=(15, 5), subplot_kw={"projection": "mollweide"}
            )
            axes = np.atleast_1d(axes)
            for pol in range(core.npol):
                # Create the mollview plot in the corresponding subplot
                plt.axes(axes[pol])
                hp.mollview(
                    lens_map[pol],
                    title=f"Simulation {sim}, Polarization {pol}",
                    hold=True,
                )

            plot_path = os.path.join(
                core.plot_dir, f"{core.sjob}_{core.base_name}_lensed.png"
            )
            plt.savefig(plot_path)
    else:
        logger.debug("Getting patches")
        patches = np.empty(
            (core.nsims, core.npatches, core.npol, core.nside, core.nside),
            dtype=core.r_dtype,
        )

        if core.use_pols:
            # need to add a zero for the spin-2 component
            alms = np.concatenate((alms, np.zeros((core.nsims, 1, core.nelem))), axis=1)

        for sim in tqdm(range(core.nsims), desc="Patching", total=core.nsims):
            car_map = curvedsky.alm2map(
                alms[sim], fs_map, copy=True, nthread=core.n_cpus
            )

            for i in range(core.npatches):
                patches[sim, i] = car_map.project(patch_shapes[i], patch_wcss[i])[: core.npol]  # type: ignore
        logger.debug("Done")
    logger.info("Done!")

    # lets plot the patches
    if core.is_main_job:
        sim = core.rng.integers(core.nsims)
        for pol in range(core.npol):
            patch_file = os.path.join(
                core.plot_dir, f"{core.sjob}_{core.base_name}_patch[{sim},{pol}].png"
            )
            plot_patches(patches[sim, :, pol], save_file=patch_file)

    logger.info("Saving data")

    sdata = {}
    sdata["alm"] = alms
    sdata["fnl"] = fnls
    sdata["patch"] = patches
    if core.lensing:
        sdata["alm_lensed"] = alm_lensed
    save_data(core.file_partial, sdata, remove_if_exists=True)

    if core.is_main_job:
        sim = core.rng.integers(core.nsims)
        for pol in range(core.npol):
            filebase = os.path.join(
                core.plot_dir, f"{core.sjob}_{core.base_name}_alm[{sim},{pol}].png"
            )

            # plot the complete alms with noise
            plot_cl_alm(
                alms[sim, pol],
                save_file=filebase,
                plot_camb=True,
                camb_cls=core.c_ells[pol],
                plot_noise=False,
                plot_full_camb=True,
                camb_noise=core.noise_ell[pol],
                camb_beam=core.beam_ell[pol],
            )

    logger.info("Finished %s!", core.sjob)


if __name__ == "__main__":
    sys.exit(main())
