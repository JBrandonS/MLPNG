import sys
import logging
import os
from functools import partial

import healpy as hp
import lenspyx
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from pixell import curvedsky, enmap, reproject
from tqdm.auto import tqdm

from . import Core
from .utils import load_data, save_data, setup_logging
from .utils.plots import plot_cl, plot_cl_map, plot_patches

logger = setup_logging(__name__, level=logging.DEBUG)


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
    nside,
    r_dtype,
    alm,
    fnl,
    cl_phi,
    plot=False,
):
    """Uses lenspyx to generate and cut the lensed flat maps"""
    geom_info = ("healpix", {"nside": nside})

    # get the synalm from the lensing potential cl's
    plm = lenspyx.utils_hp.synalm(cl_phi, lmax=max_l, mmax=None)

    # transform the lensing potential into spin-1 deflection field
    fl = np.sqrt(np.arange(max_l + 1) * np.arange(1, max_l + 2))
    dlm = lenspyx.utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

    # get the lensed map
    lens_map = lenspyx.alm2lenmap(alm, dlm, geometry=geom_info)

    # convert the map into a pixell map to allow for projection
    pixell_map = reproject.healpix2map(lens_map, fs_shape, fs_wcs, lmax)

    # cut the patches
    patches = []
    for i in range(npatches):
        patch = pixell_map.project(pshapes[i], pwcs[i])
        patches.append(patch)

    if plot:
        plot_dir = os.path.join(plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)

        map2hp = reproject.map2healpix(pixell_map, lmax)
        cls = hp.anafast(map2hp, lmax=lmax, use_pixel_weights=True)

        # reshape for when we are only doing a single pol
        if len(np.shape(cls)) == 1:
            cls = [cls]

        for pol in range(np.shape(alm)[0]):
            base = os.path.join(plot_dir, f"{base_name}_{pol}_{fnl[0]:.2f}")
            moll_path = base + "_fullsky.png"
            hp.mollzoom(map2hp[pol], title=f"pol: {pol}, fnl: {fnl[0]}")
            plt.savefig(moll_path)
            plt.close()

            map_path = base + "_lenspyx_cl_map.png"
            plot_cl(
                cls[pol],
                lmax,
                plot_camb=True,
                c_ells=c_ells[:, pol],
                save_file=map_path,
            )

            comp_path = base + "_lenspyx_cl_comp.png"
            geom = lenspyx.get_geom(geom_info)
            unl_map = geom.alm2map(
                alm[pol], lmax, None, nthreads=len(os.sched_getaffinity(0))
            )
            hp.mollzoom(
                lens_map[pol] - unl_map,
                title=f"lensed - unlensed, pol: {pol}, fnl: {fnl[0]}",
            )
            plt.savefig(comp_path)
            plt.close()

    # we want the shape to be (pol, patchs, nside, nside)
    result = np.transpose(patches, (1, 0, 2, 3))

    # need to get rid of the last pol, which is not used
    return result[:2]


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
    car_map = curvedsky.alm2map(alms, fs_map, spin=[0, 0])

    patches = []
    for i in range(npatches):
        patch = car_map.project(pshapes[i], pwcs[i])  # type: ignore
        patches.append(patch)

    if plot:
        plot_dir = os.path.join(plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)

        map2hp = reproject.map2healpix(car_map, lmax, spin=[0, 0])
        for pol in range(map2hp.shape[0]):
            hp.mollview(map2hp[pol], min=-650.0, max=650, title=f"fnl = {fnl}")
            plt.savefig(
                os.path.join(plot_dir, f"{base_name}_{fnl}_pol{pol}_fullsky.png")
            )
            plt.close()

            map_path = os.path.join(
                plot_dir, f"{base_name}_{fnl}_pol{pol}_pixell_cl_map.png"
            )
            plot_cl_map(
                car_map[pol],
                fs_wcs,
                lmax,
                plot_camb=True,
                c_ells=c_ells[:, pol],
                save_file=map_path,
            )

    # we want the shape to be (pol, patchs, nside, nside)
    result = np.transpose(patches, (1, 0, 2, 3))
    return result


def get_fs_patch_geo(core):
    """Generates the patch geometry using pixell"""
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
    return fs_shape, fs_wcs, fs_map, patch_shapes, patch_wcss


def main():
    core = Core()

    if not core.force_gen and os.path.isfile(core.patch_file):
        logger.warning("Found completed patch file, skipping patch generation")
        sys.exit(0)

    # Load in the alm data, do this first to crash fast if data is not found
    if os.path.isfile(core.alm_file_partial):
        # start with the partial file first as we can then keep a full dataset while
        # regenerating. This is useful for when we are rerunning a large number of sims

        logger.info(f"Loading alms from partial file {core.alm_file_partial}")
        ldata = load_data(core.alm_file_partial, ["alm", "fnl"])

        # for partial files, we can just start at 0
        start_idx = 0
    elif os.path.isfile(core.alm_file):
        logger.info(f"loading alms from completed file {core.alm_file}")
        ldata = load_data(core.alm_file, ["alm", "fnl"])

        # if using a completed file, we need to adjust the start index for generation
        start_idx = core.nsims * (int(core.job_array_index) - 1)  # we start at 1
        logger.info(
            "Using sims (%d, %d) out of %d",
            start_idx,
            start_idx + core.nsims,
            ldata["alm"].shape[0],
        )
    else:
        logger.fatal(
            f"No alms found. Checked {core.alm_file_partial} and {core.alm_file}, "
            "please run almgen.py first or check configuration"
        )
        sys.exit(1)

    # and read in our alms and almngs, since ldata is a h5 dataset these are not in memory
    alms = ldata["alm"]
    fnls = ldata["fnl"]

    ## Start the patch generation

    # create the array to store the patches
    patches = np.empty(core.patch_shape, dtype=core.r_dtype)

    # We setup an array with all our possible arguments to pass to the function
    # if we are using a completed alm file, we offset our sim index by the start index
    sim_arr = [start_idx + s for s in range(core.nsims)]

    # We use joblib.parallel to generate the patches in parallel
    # by default (temp_folder=None) this will use a ram disk /dev/shm
    # if the data files are larger than the available memory, about 1TB, it will error
    # so we give it a temp folder to use, which wont have that problem
    temp_folder = os.environ.get("SCRATCH", None)
    logger.debug(f"Using temp folder for patch generation: {temp_folder}")

    # lets get our generator using parallel, return as generator so we consume memory as we go
    parallel = Parallel(n_jobs=-1, return_as="generator", temp_folder=temp_folder)

    # we also setup the cutPatches function to use either pixell or lenspyx
    # depending on if we are doing lensing or not, and provide a lot of
    # arguments that are needed and will stay constant
    # we cannot abuse the python scope here since these will need to be pickled
    common_settings = [
        core.lmax,
        core.plot_dir,
        core.base_name,
        core.npatches,
        core.c_ells,
        *get_fs_patch_geo(core),
    ]
    if core.lensing:
        max_l = core.cosmo_params["max_l"]
        cl_phi = core.cosmo._camb_data.get_lens_potential_cls(  # type: ignore
            max_l, CMB_unit="muK", raw_cl=True
        )

        patch_generator = parallel(
            delayed(cutSqPatches_lenspyx)(
                *common_settings,
                max_l,
                core.nside,
                core.r_dtype,
                alms[sim],
                fnls[sim],
                cl_phi[:, 0],  # only use PP
                plot=core.is_main_job and sim == 0,
            )
            for sim in sim_arr
        )
    else:
        patch_generator = parallel(
            delayed(cutSqPatches_pixell)(
                *common_settings,
                alms[sim],
                fnls[sim],
                plot=core.is_main_job and sim == 0,
            )
            for sim in sim_arr
        )

    # Get our data from the generator, takes a long time
    for idx, result in enumerate(
        tqdm(patch_generator, desc="patch progress", total=len(sim_arr))
    ):
        patches[idx] = result

    # Save the data

    logger.info("Saving data")
    os.makedirs(core.patch_dir, exist_ok=True)

    # remove the partial file if it exists
    if os.path.isfile(core.patch_file):
        logger.info("Removing stale data file: %s", core.patch_file)
        os.remove(core.patch_file)

    sdata = {}
    sdata["fnl"] = fnls
    sdata["patch"] = patches
    save_data(core.patch_file, sdata)

    if core.is_main_job:
        plot_dir = os.path.join(core.plot_dir, "patchgen")
        os.makedirs(plot_dir, exist_ok=True)

        # lets plot the patches from a random sim
        sim = core.rng.integers(core.nsims)
        for pol in range(core.npol):
            plot_file = os.path.join(plot_dir, f"{core.base_name}_pol{pol}_patches.png")
            plot_patches(patches[sim, pol], core.npatches, save_file=plot_file)

    logger.info("Done with Generation!")


if __name__ == "__main__":
    sys.exit(main())
