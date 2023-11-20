# %%
import os
import sys
import math
import datetime
import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from functools import partial
from joblib import Parallel, delayed

import healpy as hp
import camb

from ksw import Cosmology, Data
from ksw.radial_functional import radial_func

from tqdm.auto import tqdm
from pixell import enmap, lensing, curvedsky, reproject

from utils import (
    load_data,
    save_data,
    save_plt,
    get_radii,
    plot_cl_alm,
    plot_cl_map,
)
from config import SimConfig
import lenspyx

# %matplotlib inline


def vp(*args, **kwargs):
    """prints arguments with a timestamp if verbose is set, use like print()"""
    if s.verbose:
        print(f"{datetime.datetime.now()}:", *args, **kwargs)


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
        nthreads=s.settings["nthreads_sim"],
        verbose=s.debug,
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
        plt.savefig(s.plot_dir + "/" + s.base_name + "_" + str(fnl) + "_fullsky.png")

        #     This code seems to randomly crash within the curvedsky.map2alm call
        plot_cl_map(car_map, fs_wcs, s, c_ells=c_ells, save_name=f"{s.name}-{fnl}-pfs")

    patches = []
    for i in range(s.npatches):
        patch = car_map.project(pshapes[i], pwcs[i])  # type: ignore
        patches.append(patch)

    return patches


def interpolate_ells(func, ells_sparse, ls, axis=1):
    """Our interpolation function to go from space to dense ells."""
    return CubicSpline(ells_sparse, func, axis)(ls)


def generate_almngs():
    """This code completely calculates, and saves, the alms and almngs."""
    # delta_phi = 2 * np.pi**2 * s.cosmo_params["As"] / (tr_k ** 3) * 3/5 * np.sqrt(1/2)
    # delta_phi *= (tr_k / s.cosmo_params["pivot_scalar"])**((s.cosmo_params["ns"]-1))

    A = (3 / 5) ** 2 * 2 * np.pi**2 * s.cosmo_params["As"]
    delta_phi = (tr_k) ** ((s.cosmo_params["ns"] - 1)) / (tr_k**3)

    f_k = np.ones((len(tr_k), 2), dtype=s.r_dtype) * 5 / 3
    # f_k[:, 0] = 1                           # f_k for alpha
    f_k[:, 1] *= A * delta_phi  # f_k for beta

    rad = radial_func(f_k, tr_ell_k, tr_k, radii, tr_ells)

    alpha_ell = rad[:, :, :, 0]
    alpha_l = np.concatenate(np.array([interpolate_ells(alpha_ell, tr_ells, s.ells)]))
    alpha_l = np.ascontiguousarray(alpha_l)

    beta_ell = rad[:, :, :, 1]
    c_ells_new = c_ells["c_ell"][tr_ells, : s.npol]
    div = beta_ell / c_ells_new[np.newaxis, :, :]

    bl_div_cl = np.concatenate(np.array([interpolate_ells(div, tr_ells, s.ells)]))
    bl_div_cl = np.ascontiguousarray(bl_div_cl)

    # Each alm takes ~30Mb at 1024. This is fast enough we don't need to parallelize even for very large datasets
    alms = np.array(
        [
            ksw_data.compute_alm_sim(s.lensing)
            for _ in tqdm(range(s.nsims), desc="a_lm progress")
        ],
        dtype=s.c_dtype,
    )

    # Make sure we dont get error from the beam_ell being a vector
    beam_ell_2d = np.atleast_2d(beam_ell)
    # KSW expects the alms to be coevolved with the beam
    for i in range(s.nsims):
        for j in range(s.npol):
            alms[i, j] = hp.almxfl(alms[i, j], beam_ell_2d[j] ** -1)

    sdata = {}
    sdata["alm"] = alms
    # only save 1 copy of the settings in the alms
    if s.job_array_index is None or s.job_array_index == 1:
        sdata["settings"] = s.settings
    save_data(s.alm_file_nc, sdata, verbose=s.verbose)

    # if s.debug:
    #     random_indices = [(randint(s.nsims), randint(s.npol)) for _ in range(1)]
    #     for sim, pol in random_indices:
    #         plot_cl_alm(
    #             alms[sim, pol], s, c_ells=c_ells, save_name=f"{s.name}-alm_{sim}-{pol}"
    #         )

    vp("Starting almng...")
    sim_data = np.zeros((s.nsims, s.npol, s.nelem), dtype=s.c_dtype)
    for i in range(s.nsims):
        for pol in range(s.npol):
            # create a generator
            alm_gen = Parallel(
                n_jobs=s.settings["nthreads_alm"], verbose=1, return_as="generator"
            )(
                delayed(get_alm)(
                    alms[i, pol],
                    bl_div_cl[ri, :, pol],
                    alpha_l[ri, :, pol],
                    radii[ri],
                    drs[ri],
                )
                for ri in range(len(drs))
            )

            # consume
            for alm in alm_gen:
                sim_data[i, pol] += alm

        # if s.debug:
        # plot_cl_alm(sim_data[i, pol].copy(), s, plt_camb=False, save_name=s.base_name + "_get_alm_plot_complete")

    save_data(s.alm_file_nc, {"almng": sim_data}, verbose=s.verbose)

    # if s.debug:
    #     plot_cl_alm(sim_data[0, 0], s, c_ells=c_ells, save_name=f"{s.name}-almng")

    os.replace(s.alm_file_nc, s.alm_file_partial)


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


if __name__ == "__main__":
    # %% [markdown]
    # # Data Generator
    #
    # This code primarily generates non-gaussian cmb maps. These get stored in a data file with the fnls, and patches.
    #
    # The full-sky maps are generated using the method discussed in [CMB lensing and primordial non-gaussianity](https://arxiv.org/abs/0905.4732), where we find (eq. 6)
    # $$
    # a_{\ell m} = a_{\ell m}^{{G}} + f_{NL}^X a_{\ell m}^{NG}
    # $$
    # and generated the full sky map from the $a_{\ell m}$.
    #
    # Most of this code is to calculate the term (eq. 27)
    #
    # $$
    # a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]
    # $$
    #
    # and
    #
    # $$
    # \alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)
    # $$
    #
    # $$
    # \beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)
    # $$
    #
    # $$
    # B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}
    # $$
    #
    # where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions
    #
    # ---

    # %%

    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings/settings.json"
    s = SimConfig(config_file)

    # here we setup camb since it is needed for the sims in both the alm generation
    # and patch generation
    camb_params_obj = camb.set_params(**s.cosmo_params, verbose=s.verbose)
    cosmo = Cosmology(camb_params_obj, verbose=s.verbose)
    # Additional settings here, i.e.
    # cosmo._setattr_camb('ns', 0.9624, subclass='InitPower')
    cosmo.compute_transfer(s.cosmo_params["max_l"], verbose=s.verbose)
    cosmo.compute_c_ell()

    noise_ell, beam_ell = s.get_noise_beam()
    ksw_data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

    mask = tr_ells <= s.lmax
    tr_ell_k = tr_ell_k[mask]
    tr_ells = tr_ells[mask]

    radii, drs = get_radii(s.settings["r_min"], s.settings["r_max"])

    # here we load in the alms either from a complete, combined, file or individual
    # if neither are found we generate the alms
    if os.path.isfile(s.alm_file_complete) and not s.settings["force_alm_gen"]:
        vp("Found completed alms file, skipping alm generation")
        alm_file = s.alm_file_complete
    elif os.path.isfile(s.alm_file_partial) and not s.settings["force_alm_gen"]:
        vp(f"Found partial alms file {s.alm_file_partial}, skipping alm generation")
        alm_file = s.alm_file_partial
    else:
        alm_file = s.alm_file_nc

        if os.path.isfile(s.alm_file_nc):
            vp("Removing stale alm file", s.alm_file_nc)
            os.remove(s.alm_file_nc)

        vp("Generating new alms")
        generate_almngs()
        alm_file = s.alm_file_partial

    # Just load everything into memory right now, shouldn't be much of a problem until >10k sims
    ldata = load_data(alm_file, ["alm", "almng"], verbose=s.verbose)
    alms = ldata["alm"]
    almngs = ldata["almng"]

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

    vp("Generating patches")
    vp(alms.shape, almngs.shape)
    fnls = uniform(
        s.settings["fnl_range"][0], s.settings["fnl_range"][1], (s.nsims, s.ndup)
    )
    patches = np.empty(
        (s.nsims * s.ndup, s.npol, s.npatches, s.nside, s.nside), dtype=s.r_dtype
    )
    for i in range(s.nsims):
        for j in range(s.ndup):
            for pol in range(s.npol):
                patches[i * s.ndup + j, pol] = np.array(
                    cutPatches(alms[i, pol], fnls[i, j], almngs[i, pol])
                )
    vp("Done!")

    # Save data
    sdata = {}
    sdata["fnls"] = np.atleast_1d(fnls)
    sdata["patches"] = np.array(patches)
    if s.job_array_index is None or s.job_array_index == 1:
        # we only want one copy of the settings
        sdata["settings"] = s.settings
    if os.path.isfile(s.data_file_nc):
        vp("Removing stale data file", s.data_file_nc)
        os.remove(s.data_file_nc)

    save_data(s.data_file_nc, sdata, verbose=s.verbose)
    os.replace(s.data_file_nc, s.data_file_complete)

    # %%
    vp("Done with Generation!")

    # below just generates a nice graph, possibly duplicating the patches
    # this only runs once per sim and only if debug = True
    if s.debug and (s.job_array_index is None or s.job_array_index == 1):
        nplots = 10
        random_indices = [
            (randint(s.nsims), randint(s.npol), randint(s.npatches))
            for _ in range(nplots)
        ]
        grid_size = math.isqrt(len(random_indices))
        if grid_size**2 < len(random_indices):
            grid_size += 1

        # Create the grid of subplots
        fig, axs = plt.subplots(
            grid_size, grid_size, sharex=True, sharey=True, figsize=(10, 10)
        )

        # If there's only one plot, put it in a list within a list to emulate a 2D list
        if grid_size == 1:
            axs = [[axs]]

        # Iterate over the random_indices
        for idx, (i, p, n) in enumerate(random_indices):
            # Compute the subplot coordinates
            row = idx // grid_size
            col = idx % grid_size
            # Plot the image
            axs[row][col].imshow(patches[i, p, n])

        # Hide the remaining unused subplots if any
        if len(random_indices) < grid_size * grid_size:
            for idx in range(len(random_indices), grid_size * grid_size):
                row = idx // grid_size
                col = idx % grid_size
                axs[row][col].axis("off")

        plt.title("Sample Patches")
        save_plt(s.plot_dir, f"{s.name}-sample_patches")
        plt.show()
