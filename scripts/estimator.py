import os
import sys
import numpy as np

import healpy as hp
import camb
import datetime

from ksw import Shape, KSW, Cosmology, Data
from astropy import units as u

from utils import load_data, save_data, get_radii, load_single_data
from config import SimConfig

# This will still crash on mainframe if MPI fails to start correctly....
# but try except doesn't work for some reason, and makes completion error
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def alm_loader(str_idx):
    """Loads in a single alm given a int in string form. Used inside the KSW code."""
    idx = int(str_idx)
    alm = load_single_data(s.alm_file_complete, "alm", idx, verbose=s.verbose)
    almng = load_single_data(s.alm_file_complete, "almng", idx, verbose=s.verbose)
    fnl = load_single_data(s.data_file_nc, "fnls", idx, verbose=s.verbose)
    return alm + fnl * almng


def vp(*args, **kwargs):
    """
    Printer helper, adds rank and time to the log and only prints if verbose is enabled.
    Use like print()
    """
    if s is None or rank is None:
        print("FIXME: cannot print here.")
        return

    if s.verbose and rank == 0:
        print(f"{rank} | {datetime.datetime.now()} =>", *args, **kwargs, flush=True)


if __name__ == "__main__":
    # Loads in our settings file, defaulting to settings/settings.json if no argument was provided
    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings/settings.json"
    s = SimConfig(config_file, print_settings=(rank == 0))

    # init camb and setup the reduced bispecturm to local
    vp("Running camb")
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"], verbose=s.verbose)
    cosmo.compute_c_ell()

    radii, drs = get_radii(s.settings["r_min"], s.settings["r_max"])
    loc_shape = Shape.prim_local(
        ns=s.cosmo_params["ns"], pivot=s.cosmo_params["pivot_scalar"]
    )
    cosmo.add_prim_reduced_bispectrum(loc_shape, radii)
    vp("done")

    noise_ell, beam_ell = s.get_noise_beam()
    data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
    icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

    # TODO: double check this is correct
    if s.disable_noise:

        def beam(alm):
            return alm  # hp.sphtfunc.smoothalm(alm, fwhm=0, pol=False)

    else:
        beam_width = s.settings["beam_width"] * u.arcmin
        beam_width_rad = beam_width.to_value(u.radian)

        def beam(alm):
            return hp.sphtfunc.smoothalm(alm, fwhm=beam_width_rad)

    ksw = KSW(
        cosmo.red_bispectra,
        icov,
        beam,
        s.lmax,
        s.polarizations,
        precision="double" if s.double_precision else "single",
    )

    theta_batch = 25  # nelem // 10000
    # ksw_mc_file = os.path.join(s.data_dir, 'kswmc_'+s.data_str)
    # if os.path.exists(ksw_mc_file):
    #     vp('Loading MC from file:', ksw_mc_file)
    #     ksw.start_from_read_state(ksw_mc_file, comm)
    # else:

    # 100 should be ~1%, so we take a random 100 for initializing the KSW estimator
    alm_strs = np.arange(s.total_sims).astype(str)
    if s.total_sims > 100:
        alm_strs = np.random.choice(alm_strs, size=100, replace=False)

    ksw.step_batch(alm_loader, alm_strs, comm, verbose=False, theta_batch=theta_batch)

    # Disabling for now
    # if rank == 0:
    #     ksw.write_state(ksw_mc_file, comm)

    vp("Computing estimates")
    fisher = ksw.compute_fisher()
    alm_strs = np.arange(s.total_sims).astype(str)
    estimates = ksw.compute_estimate_batch(
        alm_loader,
        alm_strs,
        comm,
        verbose=s.verbose,
        fisher=fisher,
        theta_batch=theta_batch,
    )
    vp("done")

    # save data
    if rank == 0:
        sdata = {}
        sdata["fisher"] = np.atleast_1d(fisher)
        sdata["estimates"] = estimates

        fnls = load_data(s.data_file_nc, ["fnls"], verbose=s.verbose)["fnls"]
        sdata["errors"] = (estimates - fnls) / fnls

        save_data(s.data_file_nc, sdata, verbose=s.verbose)
        os.replace(s.data_file_nc, s.data_file_complete)

    vp("Finished", rank, "!")
