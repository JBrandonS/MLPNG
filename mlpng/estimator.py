import logging
import os
import sys

import camb
import healpy as hp
import numpy as np
from astropy import units as u
from tensorflow.python.ops.gen_array_ops import rank_eager_fallback
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI
import h5py

from utils import Config, save_data, setup_logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# fix healpy logging because we will get a lot of info
logging.getLogger("healpy").setLevel(logging.WARNING)

# remove astropy warning about verbose that I can't change
logging.getLogger("astropy").setLevel(logging.ERROR)


def alm_loader(str_idx):
    """Loads in a single alm given a int in string form. Used inside the KSW code."""
    idx = int(str_idx)
    dup_idx = idx % s.ndup
    idx = idx // s.ndup
    pol = 0

    logging.info(
        f"Loading alm {idx} with dup_idx {dup_idx}, alms shape {alms_from_file.shape} and almngs shape {almngs_from_file.shape}"
    )

    alm = np.array(alms_from_file[idx, pol])
    almng = np.array(almngs_from_file[idx, pol])
    fnl = fnls[idx, dup_idx]

    logging.debug("idx: %s, dup_idx: %s, fnl: %s", idx, dup_idx, fnl)

    alm = remove_mono_dipole(alm)
    almng = remove_mono_dipole(almng)

    return alm + fnl * almng


def alm_step_loader(idx):
    """
    Used in the KSW step code to init the MC
    needs a gaussian realization of signal + noise
    """
    logger.debug("Loading alm step %s", idx)
    return data.compute_alm_sim(s.lensing)


def remove_mono_dipole(alm):
    """
    Remove the monopole and dipole terms from the alms.
    Note that we do not need -m's due to symmetry
    """
    lmax = hp.Alm.getlmax(len(alm))
    alm[hp.Alm.getidx(lmax, 0, 0)] = 0.0  # Remove monopole
    alm[hp.Alm.getidx(lmax, 1, 0)] = 0.0  # Remove dipole
    alm[hp.Alm.getidx(lmax, 1, 1)] = 0.0  # Remove dipole
    return alm


def compute_icov_ell(N, b):
    S_ell = cosmo._camb_data.get_cmb_power_spectra(
        cosmo.camb_params, s.lmax, ["total"], "muK", True
    )["total"][:, 0]
    b_inv = 1 / b
    return (1 / (S_ell + b_inv * N * b_inv))[None, :]


if __name__ == "__main__":
    is_main_comm = rank == 0
    logger = setup_logging(f"ksw estimator {rank}", level=logging.INFO if is_main_comm else logging.WARNING)
    s = Config(sys.argv[1], print_settings=is_main_comm)

    # init camb and setup the reduced bispecturm to local
    logger.info("Running camb")
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"], verbose=is_main_comm)
    cosmo.compute_c_ell()
    logger.info("done starting camb")

    # create the local shape
    loc_shape = Shape.prim_local(s.cosmo_params["ns"], s.cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, s.radii)

    # setup the data and get our icov object
    noise_ell, beam_ell = s.noise_beam
    data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
    icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

    # generate our beam functioned based on noise
    beam_width_rad = s.beam_width.to_value(u.radian) if not s.disable_noise else 0

    def beam(alm):
        if beam_width_rad == 0:
            return alm

        return hp.sphtfunc.smoothalm(alm, fwhm=beam_width_rad, inplace=False)

    ksw = KSW(
        cosmo.red_bispectra,
        icov,
        beam,
        s.lmax,
        s.polarizations,
        precision="double" if s.double_precision else "single",
    )

    # need a list of str arguments to pass into the alm_loader
    alm_strs = np.arange(s.total_sims).astype(str)

    # open the files for reading
    alm_h5_file = h5py.File(s.alm_file_complete, "r", swmr=True, locking=False)

    # these are not fully loaded into memory
    alms_from_file = alm_h5_file["alm"]
    almngs_from_file = alm_h5_file["almng"]

    # lazy load the fnls
    fnls = h5py.File(s.data_file_nc, "r", swmr=True, locking=False)["fnls"]

    logger.debug(
        f"alms: {alms_from_file.shape}, almngs: {almngs_from_file.shape}, fnls: {fnls.shape}"
    )

    # check for existing ksw state
    use_mc_file = False  # just a quick disable
    ksw_mc_file = os.path.join(s.data_dir, "kswmc_" + s.data_str)
    if use_mc_file and os.path.exists(ksw_mc_file):
        logger.info("Loading KSW state from %s", ksw_mc_file)
        ksw.start_from_read_state(ksw_mc_file, comm)
    else:
        logger.info("No KSW state found, running KSW step")
        # we dont need to step through all the alms to setup the mc
        # so this saves a lot of time
        alm_strs_i = np.arange(1000) if s.total_sims > 1000 else alm_strs
        ksw.step_batch(alm_step_loader, alm_strs_i, comm, is_main_comm)
        logger.debug("Done with KSW step")

        if use_mc_file and is_main_comm:
            ksw.write_state(ksw_mc_file, comm)

    logger.info("Computing estimates")
    fisher = ksw.compute_fisher()
    estimates = ksw.compute_estimate_batch(
        alm_loader, alm_strs, comm, verbose=is_main_comm, fisher=fisher
    )
    logger.info("done")

    # compute isotropic fisher
    icov_ell = compute_icov_ell(noise_ell, beam_ell)
    fisher_iso = ksw.compute_fisher_isotropic(icov_ell, comm=comm)

    # save data
    if is_main_comm:
        sdata = {}

        # fnls = load_data(s.data_file_nc, ["fnls"])["fnls"]
        fnls = fnls.ravel()
        est_length = estimates.shape[0]
        snr = (estimates - fnls[est_length]) * np.sqrt(fisher)

        sdata["fisher"] = np.atleast_1d(fisher)
        sdata["fisher_iso"] = np.atleast_1d(fisher_iso)
        sdata["estimates"] = estimates
        sdata["errors"] = snr

        save_data(s.data_file_nc, sdata)
        os.replace(s.data_file_nc, s.data_file_complete)

    logger.info("Finished %s!", rank)
