import logging
import os
import sys

import camb
import h5py
import healpy as hp
import numpy as np
from astropy import units as u
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI
from utils import Config, save_data, setup_logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# fix healpy logging because we will get a lot of info
logging.getLogger("healpy").setLevel(logging.WARNING)

# remove astropy warning about verbose that I can't change
logging.getLogger("astropy").setLevel(logging.ERROR)


def alm_loader(str_idx):
    """Loads in a single alm given a int in string form. Used inside the KSW code."""
    idx, pol = np.unravel_index(int(str_idx), (s.nsims, s.npol))
    logging.info(
        f"Loading alm [{idx}, {pol}], alms shape {alms.shape}, fnl {fnls[idx]}"
    )
    return np.array(alms[idx, pol])


def alm_step_loader(idx):
    """
    Used in the KSW step code to init the MC
    needs a gaussian realization of signal + noise
    """
    logger.debug("Loading alm step %s", idx)
    return data.compute_alm_sim(s.lensing)


def compute_icov_ell(N, b):
    S_ell = cosmo._camb_data.get_cmb_power_spectra(
        cosmo.camb_params,
        lmax=s.lmax,
        spectra=["total"],
        CMB_unit="muK",
        raw_cl=True,
    )["total"][:, 0]
    b_inv = 1 / b
    return (1 / (S_ell + b_inv * N * b_inv))[None, :]


if __name__ == "__main__":
    is_main = rank == 0
    logger = setup_logging(
        f"estimator {rank}", level=logging.INFO if is_main else logging.ERROR
    )
    s = Config(sys.argv[1:], print_settings=is_main)

    # init camb and setup the reduced bispecturm to local
    logger.info("Running camb")
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"], verbose=is_main)
    cosmo.compute_c_ell()
    logger.info("done starting camb")

    # create the local shape
    loc_shape = Shape.prim_local(s.cosmo_params["ns"], s.cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, s.radii)

    # setup the data and get our icov object
    data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)
    icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

    # generate our beam functioned based on noise
    beam_width_rad = 0 if s.disable_noise else s.beam_width.to_value(u.radian)

    def beam(alm):
        if beam_width_rad == 0:
            return alm

        return hp.sphtfunc.smoothalm(alm, fwhm=beam_width_rad, inplace=False)

    ksw = KSW(
        cosmo.red_bispectra,
        icov,
        beam,
        s.lmax,
        s.pols,
        precision="double" if s.double_precision else "single",
    )

    # need a list of str arguments to pass into the alm_loader
    alm_strs = np.arange(s.total_sims).astype(str)

    # open the files for reading
    alm_file = h5py.File(s.alm_file, "r", swmr=True, locking=False)

    # these are not fully loaded into memory
    alms = alm_file["alm"]
    fnls = alm_file["fnl"]
    logger.debug(f"loaded alms: {alms.shape}, fnls: {fnls.shape}")

    # check for existing ksw state
    use_mc_file = True  # just a quick disable
    ksw_mc_file = os.path.join(s.alm_dir, "kswmc_" + s.base_name)
    if use_mc_file and os.path.exists(ksw_mc_file):
        logger.info("Loading KSW state from %s", ksw_mc_file)
        ksw.start_from_read_state(ksw_mc_file, comm)
    else:
        logger.info("No KSW state found, running KSW step")

        # we dont need to step through all the alms to setup the mc
        # so this saves a lot of time
        alm_strs_i = np.arange(100)
        ksw.step_batch(alm_step_loader, alm_strs_i, comm, is_main)
        logger.info("Done with KSW step")

        if use_mc_file and is_main:
            logger.info("Saving KSW state to %s", ksw_mc_file)
            ksw.write_state(ksw_mc_file, comm)

    logger.info("Computing estimates")
    fisher = float(ksw.compute_fisher())
    estimates = ksw.compute_estimate_batch(
        alm_loader, alm_strs, comm, verbose=is_main, fisher=fisher
    )
    logger.info("done")

    # compute isotropic fisher
    icov_ell = compute_icov_ell(s.noise_ell, s.beam_ell)
    fisher_iso = ksw.compute_fisher_isotropic(icov_ell, comm=comm)

    # save data
    if is_main:
        # note: the ksw code will use all reduce so we only need to worry about the main process

        # calculate the error
        fnls = fnls.ravel()
        snr = (estimates - fnls[alm_strs.astype(int)]) * np.sqrt(fisher)

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = np.atleast_1d(fisher)
        sdata["fisher_iso"] = np.atleast_1d(fisher_iso)
        sdata["estimate"] = estimates
        sdata["error"] = snr
        save_data(s.alm_file, sdata)

        logger.info(
            f"Saved data in shapes {fisher.shape}, {fisher_iso.shape}, {estimates.shape}, {snr.shape}"
        )

    logger.info("Finished %s!", rank)
