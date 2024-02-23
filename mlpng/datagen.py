import datetime
import logging
import math
import os
import sys
from functools import partial

import camb
import healpy as hp
import lenspyx
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from ksw import Cosmology, Data
from ksw.radial_functional import radial_func
from numpy.random import randint, uniform
from pixell import curvedsky, enmap, lensing, reproject
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm
from utils import Config, load_data, save_data, setup_logging
from utils.plots import plot_cl_map, plot_cl_alm, plot_patches


def get_alm(alm, bl_div_cl, alpha_l, r, dr):
    """This calculates the alms from the precalculated values"""
    Balm = hp.almxfl(alm, bl_div_cl, inplace=False)
    B = hp.alm2map(Balm, nside=s.nside, lmax=s.lmax, pol=False, inplace=False)
    inner = hp.map2alm(B**2, lmax=s.lmax, pol=False, use_pixel_weights=True)
    kernel = hp.almxfl(inner, alpha_l, inplace=False)
    return dr * r**2 * kernel


def cutSqPatches_lenspyx(s, fs_shape, fs_wcs, pshapes, pwcs, cl_phi, alm, fnl, alm_ng):
    """Uses lenspyx to generate and cut the lensed flat maps"""
    alms = alm + fnl * alm_ng

    lmax_unl = s.cosmo_params["max_l"]
    epsilon = 1e-6  # todo: option?
    geom_info = ("healpix", {"nside": s.nside})

    # Create a full sky map with lenspyx
    plm = lenspyx.utils_hp.synalm(cl_phi, lmax=lmax_unl, mmax=None)
    fl = np.sqrt(np.arange(lmax_unl + 1) * np.arange(1, lmax_unl + 2), dtype=s.r_dtype)
    dlm = lenspyx.utils_hp.almxfl(
        plm, fl, mmax=None, inplace=False
    )  # inplace breaks, for some reason

    lens_map = lenspyx.alm2lenmap(
        alms,
        dlm,
        geometry=geom_info,
        nthreads=4,
        epsilon=epsilon,
        pol=False,
    )
    pixell_map = reproject.healpix2map(lens_map, fs_shape, fs_wcs, s.lmax)

    if s.save_fullsky and (s.job_array_index is None or s.job_array_index == 1):
        hp.mollview(pixell_map.to_healpix(), min=-650.0, max=650, title=f"fnl = {fnl}")
        plt.savefig(s.plot_dir + "/" + s.base_name + "_" + str(fnl) + "_fullsky.png")

    patches = []
    for i in range(s.npatches):
        patch = pixell_map.project(pshapes[i], pwcs[i])
        patches.append(patch)

    return patches


def cutSqPatches_pixell(s, fs_shape, fs_wcs, fs_map, pshapes, pwcs, alm, fnl, alm_ng):
    """Uses pixell to generate and cut the flat sky patches, unlensed"""
    alms = alm + fnl * alm_ng

    fs_map = enmap.empty(fs_shape, fs_wcs)
    car_map = curvedsky.alm2map(alms, fs_map)

    if s.save_fullsky and (s.job_array_index is None or s.job_array_index == 1):
        hp.mollview(car_map.to_healpix(), min=-650.0, max=650, title=f"fnl = {fnl}")
        moll_path = os.join(s.plot_dir, s.base_name + f"_{fnl}_fullsky.png")
        plt.savefig(moll_path)

        map_path = os.join(s.plot_dir, s.base_name + f"_{fnl}_pfs.fits")
        plot_cl_map(car_map, fs_wcs, plot_camb=True, c_ells=c_ells, save_file=map_path)

    patches = []
    for i in range(s.npatches):
        patch = car_map.project(pshapes[i], pwcs[i])  # type: ignore
        patches.append(patch)

    return patches


def interpolate_ells(func, ells_sparse, ls, axis=1):
    """Our interpolation function to go from space to dense ells."""
    return CubicSpline(ells_sparse, func, axis)(ls)


def generate_almngs(plot=False):
    """This code completely calculates, and saves, the alms and almngs."""
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

    # Each alm takes ~30Mb at 1024. This is fast enough we don't need to parallelize even for very large datasets
    logger.info("Starting gaussian A_lm generation")
    alms = np.array(
        [
            ksw_data.compute_alm_sim(s.lensing)
            for _ in tqdm(range(s.nsims), desc="A_lm progress")
        ],
        dtype=s.c_dtype,
    )

    # Make sure we dont get error from the beam_ell being a vector
    beam_ell_2d = np.atleast_2d(beam_ell)

    # KSW expects the alms to be coevolved with the beam
    for i in range(s.nsims):
        for j in range(s.npol):
            alms[i, j] = hp.almxfl(alms[i, j], beam_ell_2d[j] ** -1)

    logger.info("Starting non-gaussian A_lm generation")
    sim_data = np.zeros((s.nsims, s.npol, s.nelem), dtype=s.c_dtype)
    for i in tqdm(range(s.nsims), desc="almng progress"):
        for pol in range(s.npol):
            # create a generator
            alm_gen = Parallel(n_jobs=s.nthreads_alm, verbose=0, return_as="generator")(
                delayed(get_alm)(
                    alms[i, pol],
                    bl_div_cl[ri, :, pol],
                    alpha_l[ri, :, pol],
                    s.radii[ri],
                    s.drs[ri],
                )
                for ri in range(len(s.drs))
            )

            # consume
            for alm in alm_gen:
                sim_data[i, pol] += alm

    logger.info("Done!")

    sdata = {}
    sdata["alm"] = alms
    sdata["almng"] = sim_data
    if s.job_array_index is None or s.job_array_index == 1:
        sdata["settings"] = s.settings

        if plot:
            i, j = np.random.randint(s.nsims), np.random.randint(s.npol)
            alm_plot = os.path.join(s.plot_dir, s.base_name + f"_alm[{i},{j}].png")
            almng_plot = os.path.join(s.plot_dir, s.base_name + f"_almng[{i},{j}].png")

            logger.info("Plotting alm and almng")
            plot_cl_alm(alms[i, j], save_file=alm_plot, plot_camb=True, c_ells=c_ells)
            # Don't add camb to the ng plots since they are a much small scale
            plot_cl_alm(sim_data[i, j], save_file=almng_plot, plot_camb=False)

    save_data(s.alm_file_nc, sdata)
    os.replace(s.alm_file_nc, s.alm_file_partial)
    return sdata


def get_fs_patch_geo():
    """Generates the patch geometry using pixell"""
    res = np.deg2rad(s.settings["patch_side_deg"] / s.nside)
    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    fs_map = enmap.empty(fs_shape, fs_wcs)

    ps_rad = np.deg2rad(s.settings["patch_side_deg"])

    patch_shapes = []
    patch_wcss = []
    for counter in np.arange(s.npatches // 2):
        # [[dec_min,ra_min],[dec_max,ra_max]]
        top = [[0, ps_rad * counter], [ps_rad, ps_rad * (counter + 1)]]
        gs, w = enmap.geometry(pos=top, res=res, proj="car")
        patch_shapes.append(gs)
        patch_wcss.append(w)

        bottom = [[-ps_rad, ps_rad * counter], [0, ps_rad * (counter + 1)]]
        gs, w = enmap.geometry(pos=bottom, res=res, proj="car")
        patch_shapes.append(gs)
        patch_wcss.append(w)
    return fs_shape, fs_wcs, fs_map, patch_shapes, patch_wcss


# helper function to process a single patch
def process_patch(i, j, pol):
    return np.array(cutPatches(alms[i, pol], fnls[i, j], almngs[i, pol]))


if __name__ == "__main__":
    # This code primarily generates non-gaussian cmb maps. These get stored in a data file with the fnls, and patches.
    # The full-sky maps are generated using the method discussed in [CMB lensing and primordial non-gaussianity](https://arxiv.org/abs/0905.4732), where we find (eq. 6)
    # $$a_{\ell m} = a_{\ell m}^{{G}} + f_{NL}^X a_{\ell m}^{NG}$$
    # and generated the full sky map from the $a_{\ell m}$.
    # Most of this code is to calculate the term (eq. 27)
    # $$a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]$$
    # and
    # $$\alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)$$
    # $$\beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)$$
    # $$B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}$$
    # where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions

    # disable some info logs from healpy that clutter the output with
    # healpy - INFO - Sigma is 0.000000 arcmin (0.000000 rad) 
    # healpy - INFO - -> fwhm is 0.000000 arcmin
    logging.getLogger("healpy").setLevel(logging.WARNING)

    s = Config(sys.argv[1])
    is_main = (
        True if s.job_array_index is not None and s.job_array_index == 1 else False
    )

    logger = setup_logging("datagen", logging.INFO if is_main else logging.ERROR)

    # here we setup camb since it is needed for the sims in both the alm generation
    # and patch generation
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    # Additional settings here, i.e.
    # cosmo._setattr_camb('ns', 0.9624, subclass='InitPower')
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    noise_ell, beam_ell = s.noise_beam
    ksw_data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

    mask = tr_ells <= s.lmax
    tr_ell_k = tr_ell_k[mask]
    tr_ells = tr_ells[mask]

    # here we load in the alms either from a complete, combined, file or individual
    # if neither are found we generate the alms
    if os.path.isfile(s.alm_file_complete) and not s.force_alm_gen:
        logger.info("Found completed alms file, skipping alm generation")
        ldata = load_data(s.alm_file_complete, ["alm", "almng"])
    elif os.path.isfile(s.alm_file_partial) and not s.force_alm_gen:
        logger.info(
            f"Found partial alm file {s.alm_file_partial}, skipping alm generation"
        )
        ldata = load_data(s.alm_file_partial, ["alm", "almng"])
    else:
        if os.path.isfile(s.alm_file_nc):
            logger.info("Removing stale alm file: %s", s.alm_file_nc)
            os.remove(s.alm_file_nc)

        logger.info("Generating new alms")
        ldata = generate_almngs(plot=is_main)
        alm_file = s.alm_file_partial

    alms = ldata["alm"]
    almngs = ldata["almng"]

    # Here we get the geometry of our patches
    # we also setup the cutPatches function to use either pixell or lenspyx
    # depending on if we are doing lensing or not, and provide a lot of
    # arguments that are needed and will stay constant
    fs_shape, fs_wcs, fs_map, patch_shapes, patch_wcss = get_fs_patch_geo()
    if s.lensing:
        cl_phi = ksw_data.cosmology._camb_data.get_lens_potential_cls(
            s.cosmo_params["max_l"], CMB_unit="muK", raw_cl=True
        )[:, 0]
        cutPatches = partial(
            cutSqPatches_lenspyx, s, fs_shape, fs_wcs, patch_shapes, patch_wcss, cl_phi
        )
    else:
        cutPatches = partial(
            cutSqPatches_pixell, s, fs_shape, fs_wcs, fs_map, patch_shapes, patch_wcss
        )

    # Here we generate the fnls
    fnls = uniform(s.fnl_min, s.fnl_max, (s.nsims, s.ndup))

    # Start the patch generation, create the array to store the patches
    patches = np.empty(
        (s.nsims, s.ndup, s.npol, s.npatches, s.nside, s.nside), dtype=s.r_dtype
    )

    # We setup an array with all our possible arguments to pass to the function
    args = [
        (i, j, pol)
        for i in range(s.nsims)
        for j in range(s.ndup)
        for pol in range(s.npol)
    ]
    # lets get our generator setup using parallel, return as generator so we consume memory as we go
    patch_generator = Parallel(n_jobs=s.nthreads_sim // 4, return_as="generator")(
        delayed(process_patch)(*arg) for arg in args
    )

    # actually gets our data from the generator, only update every 100 runs, takes a long time
    for idx, result in enumerate(
        tqdm(patch_generator, desc="patch progress", total=len(args), miniters=100)
    ):
        i, j, pol = args[idx]
        patches[i, j, pol] = result

    # Save data
    sdata = {}
    sdata["fnls"] = fnls
    sdata["patches"] = np.array(patches)

    # only save 1 copy of the settings
    if is_main:
        sdata["settings"] = s.settings

        plot_file = os.path.join(s.plot_dir, s.base_name + "_patches.png")
        plot_patches(patches, 10, save_file=plot_file)

    # remove the partial file if it exists
    if os.path.isfile(s.data_file_nc):
        logger.debug("Removing stale data file: %s", s.data_file_nc)
        os.remove(s.data_file_nc)

    save_data(s.data_file_nc, sdata)
    logger.info("Done with Generation!")
