import logging
import os
from functools import partial

import healpy as hp
import lenspyx
import matplotlib.pyplot as plt
import numpy as np
from core import Core
from joblib import Parallel, delayed
from pixell import curvedsky, enmap, reproject
from tqdm.auto import tqdm
from utils import load_data, save_data, setup_logging
from utils.plots import plot_cl, plot_cl_map, plot_patches

logger = setup_logging(__name__)


def cutSqPatches_lenspyx(
    lmax,
    plot_dir,
    base_name,
    npatches,
    c_ells,
    fs_shape,
    fs_wcs,
    fs_map,
    pshapes,
    pwcs,
    max_l,
    cl_phi,
    nside,
    r_dtype,
    alm,
    fnl,
    plot=False,
):
    """Uses lenspyx to generate and cut the lensed flat maps"""
    geom_info = ("healpix", {"nside": nside})

    # Create a full sky map with lenspyx
    plm = lenspyx.utils_hp.synalm(cl_phi, lmax=max_l, mmax=None, rlm_dtype=r_dtype)
    fl = np.sqrt(np.arange(max_l + 1) * np.arange(1, max_l + 2), dtype=r_dtype)
    dlm = lenspyx.utils_hp.almxfl(plm, fl, mmax=None, inplace=True)

    lens_map = lenspyx.alm2lenmap(
        alm,
        dlm,
        geometry=geom_info,
        nthreads=4,
        epsilon=1e-6,
        pol=True,
    )
    pixell_map = reproject.healpix2map(lens_map, fs_shape, fs_wcs, lmax)

    patches = []
    for i in range(npatches):
        patch = pixell_map.project(pshapes[i], pwcs[i])
        patches.append(patch)

    if plot:
        plot_dir = os.path.join(plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)
        moll_path = os.path.join(plot_dir, f"{base_name}_{fnl}_fullsky.png")
        map2hp = reproject.map2healpix(pixell_map, lmax)
        hp.mollview(map2hp, min=-650.0, max=650, title=f"fnl = {fnl}")
        plt.savefig(moll_path)
        plt.close()

        map_path = os.path.join(plot_dir, f"{base_name}_{fnl}_pixell_cl_map.png")
        cls = hp.anafast(map2hp, lmax=lmax, use_pixel_weights=True)
        plot_cl(cls, lmax, plot_camb=True, c_ells=c_ells, save_file=map_path)

    return patches


def cutSqPatches_pixell(
    lmax,
    plot_dir,
    base_name,
    npatches,
    c_ells,
    fs_shape,
    fs_wcs,
    fs_map,
    pshapes,
    pwcs,
    alms,
    fnl,
    plot=False,
):
    """Uses pixell to generate and cut the flat sky patches, unlensed"""
    fs_map = enmap.empty(fs_shape, fs_wcs)
    car_map = curvedsky.alm2map(alms, fs_map)

    patches = []
    for i in range(npatches):
        patch = car_map.project(pshapes[i], pwcs[i])  # type: ignore
        patches.append(patch)

    if plot:
        plot_dir = os.path.join(plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)
        map2hp = reproject.map2healpix(car_map, lmax)
        hp.mollview(map2hp, min=-650.0, max=650, title=f"fnl = {fnl}")
        plt.savefig(os.path.join(plot_dir, f"{base_name}_{fnl}_fullsky.png"))
        plt.close()

        map_path = os.path.join(plot_dir, f"{base_name}_{fnl}_pixell_cl_map.png")
        plot_cl_map(
            car_map, fs_wcs, lmax, plot_camb=True, c_ells=c_ells, save_file=map_path
        )

    return patches


def get_fs_patch_geo():
    """Generates the patch geometry using pixell"""
    ps_rad = np.deg2rad(s.patch_side_deg)
    res = ps_rad / s.nside
    fs_shape, fs_wcs = enmap.fullsky_geometry(res, proj="car")
    fs_map = enmap.empty(fs_shape, fs_wcs)

    patch_shapes = []
    patch_wcss = []
    for counter in range(s.npatches // 2):
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
    s = Core()
    if s.is_main_job:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("utils").setLevel(logging.DEBUG)

    # Load in the alm data, do this first to crash fast if data is not found
    if os.path.isfile(s.alm_file):
        logger.info(f"loading alms from completed file {s.alm_file}")
        ldata = load_data(s.alm_file, ["alm", "fnl"])

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
        ldata = load_data(s.alm_file_partial, ["alm", "fnl"])

        # for partial files, we can just start at 0
        start_idx = 0
    else:
        logger.fatal("No alms found, please run almgen.py first or check configuration")
        exit(1)

    # and read in our alms and almngs, since ldata is a h5 dataset these are not in memory
    alms = ldata["alm"]
    fnls = ldata["fnl"]

    # Here we get the geometry of our patches in a tuple
    patch_geo = get_fs_patch_geo()

    # we also setup the cutPatches function to use either pixell or lenspyx
    # depending on if we are doing lensing or not, and provide a lot of
    # arguments that are needed and will stay constant
    # we cannot abuse the python scope here since these will need to be pickled
    common_settings = [
        s.lmax,
        s.plot_dir,
        s.base_name,
        s.npatches,
        s.c_ells,
        *patch_geo,
    ]
    if s.lensing:
        max_l = s.cosmo_params["max_l"]
        cl_phi = s.cosmo._camb_data.get_lens_potential_cls( # type: ignore
            max_l, CMB_unit="muK", raw_cl=True
        )[:, 0]

        cutPatches = partial(
            cutSqPatches_lenspyx, *common_settings, max_l, cl_phi, s.nside, s.r_dtype
        )
    else:
        cutPatches = partial(cutSqPatches_pixell, *common_settings)

    ## Start the patch generation
    # create the array to store the patches
    patches = np.empty(s.patch_shape, dtype=s.r_dtype)

    # We setup an array with all our possible arguments to pass to the function
    # if we are using a completed alm file, we offset our sim index by the start index
    args = [(start_idx + s, p) for s, p in s.sim_pol]

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
        delayed(cutPatches)(alms[i, j], fnls[i, j], s.is_main_job and i == 0)
        for i, j in args
    )

    # Get our data from the generator, only update logging every 100 runs, takes a long time
    for idx, result in enumerate(
        tqdm(patch_generator, desc="patch progress", total=s.sim_pol_len)
    ):
        sim, pol = s.sim_pol[idx]
        patches[sim, pol] = result

    # remove the partial file if it exists
    if os.path.isfile(s.patch_file):
        logger.info("Removing stale data file: %s", s.patch_file)
        os.remove(s.patch_file)

    # Save data
    os.makedirs(s.patch_dir, exist_ok=True)
    sdata = {}
    sdata["fnl"] = fnls
    sdata["patch"] = patches
    save_data(s.patch_file, sdata)

    if s.is_main_job:
        plot_dir = os.path.join(s.plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, f"{s.base_name}_patches.png")
        i, j = s.rng.integers(s.nsims), s.rng.integers(s.npol)
        plot_patches(patches[i, j], 10, save_file=plot_file)

    logger.info("Done with Generation!")
