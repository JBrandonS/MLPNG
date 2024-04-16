import logging
import os

import camb
import h5py
import healpy as hp
import numpy as np
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI
import Core
from utils import save_data, setup_logging
from utils.plots import plot_histogram, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

logger = setup_logging(name=f"{__name__}_{mpi_rank}")
if mpi_root:
    logging.getLogger("Core").setLevel(logging.DEBUG)
    logging.getLogger("utils").setLevel(logging.DEBUG)

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
    logger.debug("Sending alm step %s", idx)
    return s.data.compute_alm_sim(s.lensing)


# def get_beamfunc(s):
#     l, _ = hp.sphtfunc.Alm.getlm(s.lmax)
#     sigma = s.beam_width / np.sqrt(8 * np.log(2))
#     factor = np.exp(-(l**2) * sigma**2 / 2)

#     def _beam(alm):
#         # alm -> alm exp(-l^2 sigma^2 / 2)
#         return alm * factor

#     return _beam

# def get_beamfunc(s):
#     if not s.noise:
#         return lambda alm: alm
#     else:
#         beam_ell_pre = hp.gauss_beam(s.beam_width, lmax=s.lmax, pol=False)

#         def _beam(alm):
#             return np.array([hp.almxfl(alm[0], beam_ell_pre)])

#         return _beam

# def compute_icov_ell(N, b):
#     S_ell = cosmo._camb_data.get_cmb_power_spectra(
#         cosmo.camb_params,
#         lmax=s.lmax,
#         spectra=["total"],
#         CMB_unit="muK",
#         raw_cl=True,
#     )["total"][:, 0]
#     b_inv = 1 / b
#     return (1 / (S_ell + b_inv * N * b_inv))[None, :]


if __name__ == "__main__":
    s = Core()

    # there is a major bug in the KSW code that requires the total_sims to be >= mpi_size
    assert s.total_sims >= mpi_size, "total_sims < mpi_size, lower ntasks"

    # early loading to fail fast
    alm_file = h5py.File(s.alm_file, "r", swmr=True, locking=False)

    # check for existing ksw state, if it exists, load it
    # otherwise, run the MC, can take a few hours
    use_mc_file = True  # just a quick disable
    mc_path = os.path.join(s.alm_dir, "kswmc")
    os.makedirs(mc_path, exist_ok=True)
    mc_file = os.path.join(mc_path, f"{s.base_name}.hdf5")
    if use_mc_file and os.path.exists(mc_file):
        logger.info("Loading KSW state from %s", mc_file)
        s.ksw.start_from_read_state(mc_file, mpi_comm)
    else:
        logger.info("Running KSW step")

        # we only need to step the MC about 100 time to get a good convergence
        # there is an issue with the KSW code when the total number of steps < mpi_size
        # so we just set it to mpi_size if mpi_size > 100
        idxs = range(max(100, mpi_size))

        # we dont actually need to batch the thetas, so just set it to the full amount
        thetas = int(np.floor(1.5 * s.lmax + 1))

        # step the MC, actually does the work
        s.ksw.step_batch(alm_step_loader, idxs, comm=mpi_comm, theta_batch=thetas)

        # save the mc state if we are using the mc file
        if use_mc_file and mpi_root:
            logger.info("Saving KSW state to %s", mc_file)
            s.ksw.write_state(mc_file, mpi_comm)

    fisher = float(s.ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory
    alms = alm_file["alm"]
    fnls = alm_file["fnl"]
    logger.debug(f"using alms: {alms.shape}, fnls: {fnls.shape}")

    # only do at most 1k estimates right now
    num_est = min(1000, s.total_sims)
    logger.info(
        "Computing %s estimates in %.2f batches",
        num_est,
        num_est / mpi_size,
    )
    idxs = range(num_est)
    estimates = s.ksw.compute_estimate_batch(alm_loader, idxs, comm=mpi_comm, fisher=fisher)

    if mpi_root:
        logger.info("Saving data")

        # first, need to close the existing file or we get an error
        fnls = np.array(fnls[idxs]).flatten()
        alm_file.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = [fisher]
        sdata["estimate_1k"] = True
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(s.alm_file, sdata, mode='a')

        # make and save some plots
        plot_dir = os.path.join(s.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)
        pred_file = os.path.join(plot_dir, f"{s.sjob}_{s.base_name}_preds.png")
        hist_file = os.path.join(plot_dir, f"{s.sjob}_{s.base_name}_hist.png")
        plot_predictions(fnls, estimates, fisher, save_file=pred_file)
        plot_histogram(fnls, estimates, save_file=hist_file)

    logger.info("Finished %s!", mpi_rank)
