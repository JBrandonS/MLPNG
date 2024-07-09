import logging
import os
import sys

import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import simpson
from tqdm.auto import tqdm
from itertools import product

import lenspyx
from pixell import curvedsky, enmap, reproject

from ksw.radial_functional import radial_func

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
    B = hp.alm2map(Balm, nside=nside, lmax=lmax)
    inner = hp.map2alm(B**2, lmax=lmax, use_pixel_weights=True)
    return radii**2 * hp.almxfl(inner, alpha_l)


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


def generate_alm(core, nsims=None):
    if nsims is None:
        nsims = core.nsims

    # CAMB Cls are (nell, 4), convert to (4, nell).
    totcov = core.c_ells.copy()
    totcov *= core.beam_ell**2

    # Turn into correct shape
    if not core.use_pols:
        totcov = totcov[0, :]
        totcov = totcov[np.newaxis, :]

    # add the noise
    totcov += core.noise_ell

    # ensure totcov is contiguous (all in the same block of memory) to make computations faster
    totcov = np.ascontiguousarray(totcov)

    alms = [hp.synalm(totcov, lmax=core.lmax, new=True) for _ in range(nsims)]
    return np.array(alms, dtype=core.c_dtype)[:, : core.npol]


def generate_alm_ng(core, alms):
    """
    This function calculates the non-gaussian alms using the given core and the gaussian alms.

    Parameters:
    - core: The core object containing necessary parameters and data.
    - alms: The input alm array.

    Returns:
    - alm_ng: The calculated almng array.
    """

    ells = core.ells[2:]
    c_ells = core.c_ells[: core.npol, ells].transpose()

    tr_ells = core.cosmo.transfer["ells"]
    tr_k = core.cosmo.transfer["k"]
    tr_ell_k = core.cosmo.transfer["tr_ell_k"][..., : core.npol]

    pk = core.cosmo.camb_params.primordial_power(tr_k, 0)
    delta_phi = (2 * np.pi**2) / (tr_k ** (3)) * pk

    # this will be the f(k) value to be placed in the radial function
    # first will be for alpha_ell, second will be beta_ell
    f_k = np.full((len(tr_k), 2), 1, dtype=core.r_dtype)
    f_k[:, 0] = 5 / 3
    f_k[:, 1] = 3 / 5 * delta_phi  # f_k for beta

    # the radian_func does the f_ell^X(r) = (2/pi) int k^2 dk f(k) transfer^X_ell(k) j_ell(k r),
    rad = radial_func(f_k, tr_ell_k, tr_k, core.radii, tr_ells)

    alpha_ell = rad[..., 0]
    alpha_l = interpolator(alpha_ell, tr_ells)(ells)

    beta_ell = rad[..., 1]
    beta_l = interpolator(beta_ell, tr_ells)(ells)
    bl_div_cl = beta_l / c_ells

    # This uses joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for Alm_ng generation: {temp_folder}")

    # use return_as generator which allows us to consume the results as they are generated (in order)
    parallel = Parallel(n_jobs=-1, return_as="generator", temp_folder=temp_folder)

    alm_ng = np.zeros_like(alms)
    sim_pol = list(product(range(core.nsims), range(core.npol)))
    for sim, pol in tqdm(sim_pol, total=len(sim_pol), desc="Alm_ng"):
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

        # TODO: Figure out how to do this inline without needing the full generator as required by list
        alm_ng[sim, pol] += simpson(list(generator), x=core.radii, axis=0)
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
    # if not core.force_gen:
    #     if os.path.isfile(core.file_complete):
    #         logger.warning("Found completed file, skipping alm generation")
    #         sys.exit(0)
    #     elif os.path.isfile(core.file_partial):
    #         logger.warning("Found partial file, skipping alm generation")
    #         sys.exit(0)

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

    # finally combine into the full alms and remove the monopole and dipole terms by setting them to 0
    alms = alm_l + fnls[:, None, :] * alm_ng
    alms = remove_mono_dipole(alms)

    logger.info("Done!")

    logger.info("Starting patch generation")
    logger.debug("Generating patch geometry")

    # Generate the geometry
    logger.debug("getting patch geo")
    ps_rad = np.deg2rad(core.patch_side_deg)
    res = ps_rad / core.nside

    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    fs_shape = (core.npol,) + fs_shape
    fs_map = enmap.empty(fs_shape, fs_wcs)

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
    logger.debug("done")

    if core.lensing:
        logger.debug("getting lensing cl_phi and data")
        cl_phi = core.cosmo._camb_data.get_lens_potential_cls(  # type: ignore
            core.max_l, CMB_unit="muK", raw_cl=True
        )
        plm = lenspyx.utils_hp.synalm(cl_phi[:, 0], lmax=core.max_l, mmax=None)

        # transform the lensing potential into spin-1 deflection field
        fl = np.sqrt(np.arange(core.max_l + 1) * np.arange(1, core.max_l + 2))
        dlm = lenspyx.utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

        geom_info = ("healpix", {"nside": core.nside})
        geom = lenspyx.get_geom(geom_info)
        logger.debug("done")

        logger.debug("lensing alms and getting patches")
        # convert the map into a pixell map to allow for projection
        patches = np.empty(
            (core.nsims, core.npatches, 3, core.nside, core.nside), dtype=core.r_dtype
        )
        alm_lensed = np.empty((core.nsims, 3, core.nelem), dtype=core.c_dtype)
        for sim in tqdm(
            range(core.nsims), desc="Lensing and Patching", total=core.nsims
        ):
            lens_map = lenspyx.alm2lenmap(alms[sim], dlm, geometry=geom_info, verbose=0)
            pixell_map = reproject.healpix2map(lens_map, fs_shape, fs_wcs, core.lmax)

            alm_lensed[sim, 0] = geom.map2alm(
                lens_map[0], core.lmax, core.lmax, nthreads=os.cpu_count()
            )
            alm_lensed[sim, 1:] = geom.map2alm_spin(
                lens_map[1:], 2, core.lmax, core.lmax, nthreads=os.cpu_count()
            )
            # cut the patches
            for i in range(core.npatches):
                patches[i] = pixell_map.project(patch_shapes[i], patch_wcss[i])
    else:
        logger.debug("getting patches")
        fs_map = enmap.empty(fs_shape, fs_wcs)
        patches = np.empty(
            (core.nsims, core.npatches, core.npol, core.nside, core.nside),
            dtype=core.r_dtype,
        )

        for sim in tqdm(range(core.nsims), desc="Patching", total=core.nsims):
            car_map = curvedsky.alm2map(alms[sim], fs_map, spin=[0, 0])

            for i in range(core.npatches):
                patch = car_map.project(patch_shapes[i], patch_wcss[i])  # type: ignore
                patches[sim, i] = patch
    logger.info("Done!")

    logger.info("Saving data")
    os.makedirs(core.data_dir, exist_ok=True)
    # we remove the stale nc file if it exists prior to saving the new one
    if os.path.isfile(core.file_partial):
        logger.info("Removing stale alm file: %s", core.file_partial)
        os.remove(core.file_partial)

    sdata = {}
    sdata["alm"] = alms
    sdata["fnl"] = fnls
    sdata["patch"] = patches
    if core.lensing:
        sdata["alm_lensed"] = alm_lensed
    save_data(core.file_partial, sdata)

    if core.is_main_job:
        alm_plot_dir = os.path.join(core.plot_dir, "generator")
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
            c_ells=core.c_ells[j],
            camb_noise=True,
            noise=core.noise_ell[j],
            beam_width=core.beam_width,
        )

    logger.info("Finished %s!", core.sjob)


if __name__ == "__main__":
    sys.exit(main())
