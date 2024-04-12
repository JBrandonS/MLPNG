import logging
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
from numpy.random import randint, uniform
from pixell import curvedsky, enmap, reproject
from tqdm.auto import tqdm

from .utils import Config, load_data, save_data, setup_logging
from .utils.plots import plot_cl_map, plot_patches

logger = setup_logging(__name__)

def cutSqPatches_lenspyx(s, fs_shape, fs_wcs, fs_map, pshapes, pwcs, cl_phi, alms, fnl):
    """Uses lenspyx to generate and cut the lensed flat maps"""

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

    if s.is_main_job:
        map2hp = reproject.map2healpix(pixell_map, s.lmax)
        hp.mollview(map2hp, min=-650.0, max=650, title=f"fnl = {fnl}")

        plot_dir = os.path.join(s.plot_dir, "patchgen")
        moll_path = os.path.join(plot_dir, s.base_name + f"_{fnl}_fullsky.png")
        plt.savefig(moll_path)

    patches = []
    for i in range(s.npatches):
        patch = pixell_map.project(pshapes[i], pwcs[i])
        patches.append(patch)

    return patches


def cutSqPatches_pixell(s, fs_shape, fs_wcs, fs_map, pshapes, pwcs, alms, fnl):
    """Uses pixell to generate and cut the flat sky patches, unlensed"""
    fs_map = enmap.empty(fs_shape, fs_wcs)
    car_map = curvedsky.alm2map(alms, fs_map)

    if s.is_main_job:
        map2hp = reproject.map2healpix(car_map, s.lmax)
        hp.mollview(map2hp, min=-650.0, max=650, title=f"fnl = {fnl}")

        plot_dir = os.path.join(s.plot_dir, "patchgen")
        moll_path = os.path.join(plot_dir, s.base_name + f"_{fnl}_fullsky.png")
        plt.savefig(moll_path)

        map_path = os.path.join(plot_dir, s.base_name + f"_{fnl}_pixell_cl_map.png")
        plot_cl_map(car_map, fs_wcs, plot_camb=True, c_ells=c_ells, save_file=map_path)

    patches = []
    for i in range(s.npatches):
        patch = car_map.project(pshapes[i], pwcs[i])  # type: ignore
        patches.append(patch)

    return patches


def get_fs_patch_geo():
    """Generates the patch geometry using pixell"""
    res = np.deg2rad(s.patch_side_deg / s.nside)
    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    fs_map = enmap.empty(fs_shape, fs_wcs)

    ps_rad = np.deg2rad(s.patch_side_deg)

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

    s = Config()

    # Load in the alm data, do this first to crash fast if data is not found
    if os.path.isfile(s.alm_file):
        logger.info(f"loading alms from completed file {s.alm_file}")
        ldata = load_data(s.alm_file, ["alm", "fnls"])

        # if using a completed file, we need to adjust the start index for generation
        start_idx = s.nsims * (int(s.job_array_index) - 1)  # we start at 1 rn
        logger.info(
            "Using sims (%d, %d) out of %d",
            start_idx,
            start_idx + s.nsims,
            ldata["alm"].shape[0],
        )

    elif os.path.isfile(s.alm_file_partial):
        logger.info(f"Loading alms from partial file {s.alm_file_partial}")
        ldata = load_data(s.alm_file_partial, ["alm", "fnls"])

        # for partial files, we can just start at 0
        start_idx = 0
    else:
        logger.fatal("No alms found, please run almgen.py first or check configuration")
        exit(1)

    # and read in our alms and almngs, since ldata is a h5 dataset these are not in memory
    alms = ldata["alm"]
    fnls = ldata["fnl"]

    # here we setup camb since it is needed for the sims in the patch generation
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    # get our ksw object
    ksw_data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)

    # Get the transfer data from ksw
    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

    # CAMB will use max_l to generate the transfer functions, this is more than we need
    # so we need to mask the transfer functions to the lmax we are using
    mask = tr_ells <= s.lmax
    tr_ell_k = tr_ell_k[mask]
    tr_ells = tr_ells[mask]

    # Here we get the geometry of our patches in a tuple
    patch_geo = get_fs_patch_geo()

    # we also setup the cutPatches function to use either pixell or lenspyx
    # depending on if we are doing lensing or not, and provide a lot of
    # arguments that are needed and will stay constant
    # we cannot abuse the python scope here since these will need to be pickled
    if s.lensing:
        cl_phi = ksw_data.cosmology._camb_data.get_lens_potential_cls(
            s.cosmo_params["max_l"], CMB_unit="muK", raw_cl=True
        )[:, 0]
        cutPatches = partial(cutSqPatches_lenspyx, s, *patch_geo, cl_phi)
    else:
        cutPatches = partial(cutSqPatches_pixell, s, *patch_geo)

    ## Start the patch generation
    # create the array to store the patches
    patches = np.empty((s.nsims, s.npol, s.npatches, s.nside, s.nside), dtype=s.r_dtype)

    # We setup an array with all our possible arguments to pass to the function
    # if we are using a completed alm file, we offset our sim index by the start index
    args = [(start_idx + i, pol) for i in range(s.nsims) for pol in range(s.npol)]

    # We use joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, about 1TB, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for patch generation: {temp_folder}")

    # lets get our generator using parallel, return as generator so we consume memory as we go
    patch_generator = Parallel(
        n_jobs=-1,
        return_as="generator",
        temp_folder=temp_folder,
    )(
        delayed(lambda i, pol: np.array(cutPatches(alms[i, pol], fnls[i, pol])))(*arg)
        for arg in args
    )

    # Get our data from the generator, only update logging every 100 runs, takes a long time
    for idx, result in enumerate(
        tqdm(
            patch_generator,
            desc="patch progress",
            total=len(args),
            miniters=100,
        )
    ):
        i, pol = args[idx]
        patches[i, pol] = result

    # remove the partial file if it exists
    if os.path.isfile(s.patch_file_partial):
        logger.debug("Removing stale data file: %s", s.patch_file_partial)
        os.remove(s.patch_file_partial)

    # Save data
    sdata = {}
    sdata["fnl"] = fnls
    sdata["patch"] = np.array(patches)

    # only save 1 copy of the settings
    if s.is_main_job:
        sdata["settings"] = s.settings

        plot_dir = os.path.join(s.plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, s.base_name + "_patches.png")
        plot_patches(patches, 10, save_file=plot_file)

    os.makedirs(s.patch_dir, exist_ok=True)
    save_data(s.patch_file_partial, sdata)
    os.replace(s.patch_file_partial, s.patch_file)
    logger.info("Done with Generation!")
