import sys
import logging
import os
import healpy as hp
import camb

import h5py
import numpy as np
from mpi4py import MPI

from . import Core
from .generator import generate_alm
from .utils import save_data, setup_logging
from .utils.plots import plot_histogram, plot_predictions

from ksw import KSW, Cosmology, Shape

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

# estimator will run multiple jobs per id which get sent to the same log file,
# so we only want to log the root to keep from spamming the log
if mpi_root:
    logger = setup_logging(name=f"{__name__}_{mpi_rank}", level=logging.DEBUG)
else:
    logger = setup_logging(name=f"{__name__}_{mpi_rank}", level=logging.ERROR)


def icov_func(beam, noise, c_ells):
    """
    Returned the inverse covariance of the data, used in the KSW estimator.

    Function takes (npol, nelem) alm-like complex array "alm" and returns the
    inverse-variance-weighted version of that array. Specifically:
    (B^{-1} N B^{-1} + S)^{-1} B^{-1} a, where a = B s + n, B is the beam
    and N^{-1} and S^{-1} are the inverse noise and signal covariance
    matrices, respectively.

    Parameters:
    - alm (ndarray): (npol, nelem) alm-like complex array.

    Returns:
    - ndarray: Inverse-variance-weighted version of the input array.

    """

    # get needed values and remove the mono and dipole terms
    B_inv = 1 / beam
    S = c_ells

    factor = (B_inv * noise * B_inv + S) ** (-1) * B_inv
    npol = factor.shape[0]

    def _func(alm):
        ret = np.zeros_like(alm)
        for pol in range(npol):
            ret[pol] = hp.almxfl(alm[pol], factor[pol])
        return ret

    return _func


def conv_beam_func(core, npol):  # change name
    """
    Returns a function which KSW can use to convolve the alms with the beam.
    If beam_width is 0, returns the identity function.

    Returns:
        function: A function that takes alm values and returns the convolved beam.
    """
    if core.beam_width == 0.0:
        # Dont need to bother with anything if beam_width is 0
        return lambda alm: alm

    def __beam(alm):
        # Convolve the beam with the alm values
        ret = np.zeros_like(alm)
        for pol in range(npol):
            ret[pol] = hp.almxfl(alm[pol], core.beam_ell[pol])
        return ret

    return __beam


def compute_iso_icov(core, N, b):
    # Inverse covariance matrix diagonal in ell. Unlike "icov" this should be: 1 / (S_ell + (b^{-1} N b^{-1})_ell)
    S_ell = core.cosmo._camb_data.get_cmb_power_spectra(
        core.cosmo.camb_params,
        core.lmax,
        ["total"],
        "muK",
        True,
    )[
        "total"
    ]  # TODO: fix this
    S_ell = np.transpose(S_ell, (1, 0))[: core.npol]
    b_inv = 1 / b
    return 1 / (S_ell + b_inv * N * b_inv)


def run_ksw_step(ksw, core, theta_batch, num_steps=100):
    logger.debug("Generating the ksw alms")
    alm_steps = generate_alm(core, num_steps)

    logger.debug("Done")

    def step_loader(idx):
        """
        for stepping the KSW estimator, we just generate new unique sims
        """
        logger.debug("Sending alm step %s", idx)
        return alm_steps[idx]

    logger.info("Running KSW step, num steps: %s", num_steps)
    ksw.step_batch(
        step_loader, np.arange(num_steps), comm=mpi_comm, theta_batch=theta_batch
    )
    logger.info("Finished KSW step")

    # compute the new fisher and the distance
    fisher = ksw.compute_fisher()
    icov_ell = compute_iso_icov(
        core, core.noise_ell[: core.npol], core.beam_ell[: core.npol]
    )
    fisher_iso = ksw.compute_fisher_isotropic(icov_ell)
    distance = np.abs(fisher - fisher_iso)
    logger.debug("Fisher distance: %s", distance)


def main():
    core = Core()

    # the KSW code requires the total_sims to be >= mpi_size
    # best usage would have total_sims % mpi_size == 0, but not required
    assert (
        core.total_sims >= mpi_size
    ), "total_sims < mpi_size, lower ntasks or increase sims"

    # early loading to fail fast if the file does not exist
    data = h5py.File(core.file_complete, "r", swmr=True, locking=False)

    # we need to setup the KSW here
    cosmo_params = core.cosmo_params
    cosmo = Cosmology(camb.set_params(**cosmo_params))
    cosmo.compute_transfer(core.max_l)
    cosmo.compute_c_ell()

    noise = core.noise_ell[: core.npol]
    beam = core.beam_ell[: core.npol]
    c_ells = core.c_ells[: core.npol]

    loc_shape = Shape.prim_local(cosmo_params["ns"], cosmo_params["pivot_scalar"])
    cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    icov = icov_func(beam, noise, c_ells)

    ksw = KSW(
        cosmo.red_bispectra,
        icov,
        conv_beam_func(core, core.npol),
        core.lmax,
        core.pols,
        core.precision,
    )

    # The default theta_batch size is 25, which is really small, we want to increase it
    # going too high can cause memory issues, so we will cap it at 256
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    theta_batch = min(256, theta_batch)
    logger.debug("Using theta_batch %s", theta_batch)

    # remove the file to force its recreation
    if core.force_ksw and os.path.exists(core.mc_file):
        logger.info("Removing existing KSW state")
        os.remove(core.mc_file)

    if os.path.exists(core.mc_file):
        logger.info("Loading KSW state from %s", core.mc_file)
        ksw.start_from_read_state(core.mc_file, mpi_comm)
    else:
        run_ksw_step(ksw, core, theta_batch)

        # save the mc state if we are using the mc file
        if mpi_root:
            logger.info("Saving KSW state to %s", core.mc_file)
            ksw.write_state(core.mc_file, mpi_comm)

    fisher = float(ksw.compute_fisher())
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory
    alms = data["alm_lensed"] if core.lensing else data["alm"]
    fnls = data["fnl"]

    idxs = np.arange(core.num_estimates)
    logger.info(
        "Computing %s estimates in %.2f batches",
        core.num_estimates,
        core.num_estimates / mpi_size,
    )
    if core.num_estimates % mpi_size != 0:
        logger.debug(
            "WARNING: num_estimates is not divisible by mpi_size, "
            "this will lead to uneven workloads."
        )

    def estimator_loader(idx):
        """Loads in a single alm given an idx."""
        logger.debug("Sending %s with fnl %s", idx, fnls[idx])
        return np.array(alms[idx, : core.npol])

    estimates = ksw.compute_estimate_batch(
        estimator_loader,
        idxs,
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
    )

    if mpi_root:
        logger.info("Saving data")

        fnls = np.array(fnls[idxs]).flatten()

        # alm_file is read only and we need to append to it
        # close it, so we can open in append mode
        data.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = [fisher]
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(core.file_complete, sdata, mode="a")

        # make and save some plots
        plot_dir = os.path.join(core.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)

        pred_file = os.path.join(plot_dir, f"{core.sjob}_{core.base_name}_preds.png")
        plot_predictions(fnls, estimates, fisher=fisher, save_file=pred_file)

        hist_file = os.path.join(plot_dir, f"{core.sjob}_{core.base_name}_hist.png")
        plot_histogram(fnls, estimates, save_file=hist_file)

        diff = estimates - fnls
        std_dev = np.sqrt(1 / fisher)
        sem = std_dev / np.sqrt(len(diff))
        logger.info("Standard deviation: %s, SEM: %s", std_dev, sem)
        for i in range(5):
            within = np.sum(np.abs(diff) < ((i + 1) * std_dev))
            percentage = within / len(diff) * 100
            within_error = np.sum(np.abs(diff) < ((i + 1) * (std_dev + sem)))
            percentage_error = within_error / len(diff) * 100
            logger.info(
                "%s%% are within %s standard deviations, and %s%% are within standard deviations with error.",
                percentage,
                i + 1,
                percentage_error,
            )

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
