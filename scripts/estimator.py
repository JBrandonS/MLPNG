import logging
import os

import camb
import h5py
import healpy as hp
import numpy as np
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI
from utils import Config, save_data, setup_logging
from utils.plots import plot_histogram, plot_ksw_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

logger = setup_logging(
    name=f"{__name__}_{mpi_rank}", level=logging.DEBUG if mpi_root else logging.ERROR
)
if mpi_root:
    logging.getLogger("utils.config").setLevel(logging.DEBUG)
    logging.getLogger("utils.utils").setLevel(logging.DEBUG)

logging.getLogger("healpy").setLevel(logging.ERROR)
logging.getLogger("astropy").setLevel(logging.ERROR)


def alm_loader(str_idx):
    """Loads in a single alm given a int in string form. Used inside the KSW code."""
    idx, pol = np.unravel_index(int(str_idx), (s.total_sims, s.npol))
    logger.debug("Sending alm (%s, %s) with fnl %s", idx, pol, fnls[idx, pol])
    return np.array(alms[idx, pol])


def alm_step_loader(idx):
    """
    Used in the KSW step code to init the MC
    needs a gaussian realization of signal + noise
    """
    logger.info("Sending alm step %s", idx)
    return data.compute_alm_sim(s.lensing)


def beam(alm):
    return hp.sphtfunc.smoothalm(alm, fwhm=s.beam_width)


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
    s = Config()
    assert s.total_sims >= mpi_size, "total_sims < mpi_size, lower ntasks"

    # early loading to fail fast
    alm_file = h5py.File(s.alm_file, "r", swmr=True, locking=False)

    # init camb and setup the reduced bispecturm to local
    logger.info("Setting up camb")
    camb_params_obj = camb.set_params(**s.cosmo_params)
    cosmo = Cosmology(camb_params_obj)
    cosmo.compute_transfer(s.cosmo_params["max_l"])
    cosmo.compute_c_ell()

    logger.info("Setting up KSW")
    # create the local shape
    loc_shape = Shape.prim_local(s.cosmo_params["ns"], s.cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, s.radii)
    logger.debug("Finished adding local shape")

    # setup the data and get our icov object
    data = Data(s.lmax, s.noise_ell, s.beam_ell, s.pols, cosmo)
    icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed
    ksw = KSW(cosmo.red_bispectra, icov, beam, s.lmax, s.pols, s.precision)

    # compute the isotropic fisher
    logger.info("Computing isotropic fisher")
    logger.debug("getting icov ell")
    icov_ell = compute_icov_ell(s.noise_ell, s.beam_ell)
    logger.debug("finished getting icov ell, starting fisher iso")
    fisher_iso = ksw.compute_fisher_isotropic(icov_ell, comm=mpi_comm)
    logger.info(
        "Isotropic fisher: %s, standard deviation: %s",
        fisher_iso,
        np.sqrt(1 / fisher_iso),
    )

    # check for existing ksw state, if it exists, load it
    # otherwise, run the MC, can take a few hours
    use_mc_file = True  # just a quick disable
    mc_path = os.path.join(s.alm_dir, "kswmc")
    os.makedirs(mc_path, exist_ok=True)
    mc_file = os.path.join(mc_path, f"{s.base_name}_mc")
    if use_mc_file and os.path.exists(mc_file):
        logger.info("Loading KSW state from %s", mc_file)
        ksw.start_from_read_state(mc_file, mpi_comm)
    else:
        logger.info("Running KSW step")

        # we only need to step the MC about 100 time to get a good convergence
        # there is an issue with the KSW code when the total number of steps < mpi_size
        # so we just set it to mpi_size if mpi_size > 100
        idxs = range(max(100, mpi_size))

        # we dont actually need to batch the thetas, so just set it to the full amount
        thetas = int(np.floor(1.5 * s.lmax + 1))

        # step the MC
        ksw.step_batch(alm_step_loader, idxs, comm=mpi_comm, theta_batch=thetas)

        # save the mc state if we are using the mc file
        if use_mc_file and mpi_root:
            logger.info("Saving KSW state to %s", mc_file)
            ksw.write_state(mc_file, mpi_comm)

    fisher = float(ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory
    alms = alm_file["alm"]
    fnls = alm_file["fnl"]
    logger.debug(f"using alms: {alms.shape}, fnls: {fnls.shape}")

    logger.info(
        "Computing %s estimates in %.2f batches",
        s.total_sims,
        s.total_sims / mpi_size,
    )
    idxs = range(s.total_sims)
    estimates = ksw.compute_estimate_batch(alm_loader, idxs, comm=mpi_comm, fisher=fisher)

    if mpi_root:
        logger.info("Saving data")

        # first, need to close the existing file or we get an error
        fnls = np.array(fnls[:]).flatten()
        alm_file.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = [fisher]
        sdata["fisher_iso"] = [fisher_iso]
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(s.alm_file, sdata)

        # make and save some plots
        plot_dir = os.path.join(s.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)
        pred_file = os.path.join(plot_dir, f"{s.sjob}_{s.base_name}_preds.png")
        hist_file = os.path.join(plot_dir, f"{s.sjob}_{s.base_name}_hist.png")
        plot_ksw_predictions(fnls, estimates, fisher, save_file=pred_file)
        plot_histogram(fnls, estimates, save_file=hist_file)

    logger.info("Finished %s!", mpi_rank)
