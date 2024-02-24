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

from utils import Config, load_data, save_data, setup_logging
from utils.plots import plot_cl_map, plot_patches


def cutSqPatches_lenspyx(s, fs_shape, fs_wcs, fs_map, pshapes, pwcs, cl_phi, alm, fnl, alm_ng):
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

    if (s.job_array_index is None or s.job_array_index == 1):
        map2hp = reproject.map2healpix(pixell_map, s.lmax)
        hp.mollview(map2hp, min=-650.0, max=650, title=f"fnl = {fnl}")
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

    if (s.job_array_index is None or s.job_array_index == 1):
        map2hp = reproject.map2healpix(car_map, s.lmax)
        hp.mollview(map2hp, min=-650.0, max=650, title=f"fnl = {fnl}")
        moll_path = os.join(s.plot_dir, s.base_name + f"_{fnl}_fullsky.png")
        plt.savefig(moll_path)

        map_path = os.join(s.plot_dir, s.base_name + f"_{fnl}_pixell_cl_map.png")
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


# helper function to process a single patch
def process_patch(i, pol):
    return np.array(cutPatches(alms[i, pol], fnls[i], almngs[i, pol]))


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

    s = Config(sys.argv)
    is_main = True if s.job_array_index is None or s.job_array_index == 1 else False

    logger = setup_logging("patchgen", logging.INFO if is_main else logging.ERROR)

    # Load in the alm data
    if os.path.isfile(s.alm_file):
        logger.info(f"loading alms from completed file {s.alm_file}")
        ldata = load_data(s.alm_file, ["alm", "almng"])
    elif os.path.isfile(s.alm_file_partial):
        logger.info(f"Loading alms from partial file {s.alm_file_partial}")
        ldata = load_data(s.alm_file_partial, ["alm", "almng"])
    else:
        logger.fatal("No alms found, please run almgen.py first")
        exit(1)

    # and read in our alms and almngs, since ldata is a h5 dataset these are not in memory
    alms = ldata["alm"]
    almngs = ldata["almng"]

    # here we setup camb since it is needed for the sims in the patch generation
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    noise_ell, beam_ell = s.noise_beam
    ksw_data = Data(s.lmax, noise_ell, beam_ell, s.pols, cosmo)

    c_ells = ksw_data.cosmology.c_ell["unlensed_scalar"]  # type: ignore
    tr_ell_k = ksw_data.cosmology.transfer["tr_ell_k"]
    tr_ells = ksw_data.cosmology.transfer["ells"]
    tr_k = ksw_data.cosmology.transfer["k"]

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

    # Here we generate the fnls
    fnls = uniform(s.fnl_min, s.fnl_max, s.nsims)

    # Start the patch generation, create the array to store the patches
    patches = np.empty(
        (s.nsims, s.npol, s.npatches, s.nside, s.nside), dtype=s.r_dtype
    )

    # We setup an array with all our possible arguments to pass to the function
    args = [
        (i, pol)
        for i in range(s.nsims)
        for pol in range(s.npol)
    ]

    # lets get our generator setup using parallel, return as generator so we consume memory as we go
    patch_generator = Parallel(n_jobs=-1, return_as="generator")(
        delayed(process_patch)(*arg) for arg in args
    )

    # actually gets our data from the generator, only update every 100 runs, takes a long time
    for idx, result in enumerate(
        tqdm(patch_generator, desc="patch progress", total=len(args), miniters=100)
    ):
        i, pol = args[idx]
        patches[i, pol] = result

    # remove the partial file if it exists
    if os.path.isfile(s.data_file_nc):
        logger.debug("Removing stale data file: %s", s.data_file_nc)
        os.remove(s.data_file_nc)

    # Save data
    sdata = {}
    sdata["fnls"] = fnls
    sdata["patches"] = np.array(patches)

    # only save 1 copy of the settings
    if is_main:
        sdata["settings"] = s.settings

        plot_file = os.path.join(s.plot_dir, s.base_name + "_patches.png")
        plot_patches(patches, 10, save_file=plot_file)

    save_data(s.data_file_nc, sdata)
    logger.info("Done with Generation!")
