import sys
import logging
import os
import healpy as hp
import camb

import h5py
import numpy as np
from mpi4py import MPI

from ksw import KSW, Shape, Cosmology

from . import Core
from .generator import generate_alm
from .utils import save_data, setup_logging, print_errors
from .utils.plots import plot_histogram, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

# estimator will run multiple jobs per id which get sent to the same log file,
# so we only want to log the root to keep from spamming the log
if mpi_root:
    logger = setup_logging(name=f"estimator_{mpi_rank}", level=logging.DEBUG)
else:
    logger = setup_logging(name=f"estimator_{mpi_rank}", level=logging.ERROR)


def icov_func(icov, alm):
    ret = np.zeros_like(alm)
    for pol in range(ret.shape[0]):
        ret[pol] = hp.almxfl(alm[pol].copy(), icov[pol])
    return ret


def run_ksw_step(ksw, core, theta_batch, num_steps=100):
    alm_steps = generate_alm(core, num_steps)

    def step_loader(idx):
        """for stepping the KSW estimator, we just generate new unique sims"""
        logger.debug("Sending alm step %s", idx)
        return alm_steps[idx]

    ksw.step_batch(step_loader, range(num_steps), mpi_comm, theta_batch=theta_batch)


def main():
    core = Core()

    # the KSW code requires the total_sims to be >= mpi_size
    # best usage would have total_sims % mpi_size == 0, but not required
    assert (
        core.total_sims >= mpi_size
    ), "total_sims < mpi_size, lower ntasks or increase sims"

    logger.info(
        "Computing %s estimates in %.2f batches",
        core.num_estimates,
        core.num_estimates / mpi_size,
    )
    if core.num_estimates % mpi_size != 0:
        logger.warning(
            "num_estimates is not divisible by mpi_size, "
            "this will lead to uneven workloads."
        )

    # early loading to fail fast if the file does not exist
    data_file = h5py.File(core.file_complete, "r", swmr=True, locking=False)

    # setup our cosmology and compute the c_ells
    camb_params = camb.set_params(**core.cosmo_params)
    ip = camb.initialpower.InitialPowerLaw()
    ip.set_params(As=core.cosmo_params["As"], ns=core.cosmo_params["ns"])
    camb_params.set_initial_power(ip)
    cosmo = Cosmology(camb_params)
    cosmo.compute_transfer(core.lmax)
    cosmo.compute_c_ell()
    loc_shape = Shape.prim_local(
        core.cosmo_params["ns"], core.cosmo_params["pivot_scalar"]
    )
    cosmo.add_prim_reduced_bispectrum(loc_shape, core.radii)

    if core.lensing:
        c_ells = cosmo.c_ell["lensed_scalar"]["c_ell"].T
    else:
        c_ells = cosmo.c_ell["unlensed_scalar"]["c_ell"].T
    c_ells = c_ells[:, : core.nell].copy()  # trim c_ells to the correct length

    # and lets get the icov_ell
    cov_tot = np.sqrt(c_ells.copy()) # ** 2  # core.beam_ell**2 * c_ells + core.noise_ell
    cov_tot[..., :2] = 0  # we do not want to use the mono and dipole terms

    # n = (2 * np.arange(core.nell) + 1) / 2
    # cov_tot[..., 2:] /= n[None, 2:]
    print("cov_tot", cov_tot) 
    # if core.use_te:
    #     # we need to square the matrix and invert it properly
    #     cov = np.zeros((2, 2, core.nell))
    #     cov[0, 0] = cov_tot[0]  # TT
    #     cov[1, 1] = cov_tot[1]  # EE
    #     cov[1, 0] = cov_tot[3]  # ET
    #     cov[0, 1] = cov_tot[3]  # TE
    #     icov = np.linalg.inv(cov.T).T

    #     # lets flatten this for our code, will be TT, EE, BB, TE order to match alms
    #     icov_ell = np.array(
    #         [
    #             icov[0, 0],  # TT
    #             icov[1, 1],  # EE
    #             np.zeros(core.nell),  # BB
    #             icov[1, 0],  # TE
    #         ],
    #     )
    # else:
    # to avoid a divide by zero we skip the mono and dipole terms, and b mode
    icov_ell = np.zeros_like(cov_tot)
    icov_ell[[0, 1, 3], 2:] = 1 / cov_tot[[0, 1, 3], 2:]
    icov_ell = icov_ell.copy()

    ksw = KSW(
        cosmo.red_bispectra,
        None,  # lambda a: a,  # icov_func(icov_ell, a),  # we will send the alms directly
        core.lmax,
        core.pols,
        core.precision,
    )

    # The default theta_batch size is 25, which is really small, we want to increase it
    theta_batch = int(np.floor(1.5 * core.lmax + 1)) // mpi_size
    logger.debug("Using theta_batch %s", theta_batch)

    # since we are only doing full sky right now, we do not need to use MC methods
    if core.force_ksw:  # TODO: test mc methods
        if os.path.exists(core.mc_file):
            logger.info("Loading KSW state from %s", core.mc_file)
            ksw.start_from_read_state(core.mc_file, mpi_comm)
        else:  # not tested much
            run_ksw_step(ksw, core, theta_batch)

            # save the mc state if we are using the mc file
            if mpi_root:
                logger.info("Saving KSW state to %s", core.mc_file)
                ksw.write_state(core.mc_file, mpi_comm)

        fisher = float(ksw.compute_fisher())  # type: ignore
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            # ib = np.where(core.beam_ell != 0, 1 / core.beam_ell, 0)
            icov = np.where(
                cov_tot != 0,  # + ib * core.noise_ell * ib != 0,
                1 / (cov_tot),  # + ib * core.noise_ell * ib),
                0,
            )
            icov = icov[core.pol_idxs()].copy()

        nelem = icov.shape[1]

        # Create a new array of shape (2, 2, nelem) initialized with zeros
        # icov_new = np.zeros((2, 2, nelem))

        # Fill the diagonal elements with the values from the original icov array
        # icov_new[0, 0, :] = icov[0, :]
        # icov_new[1, 1, :] = icov[1, :]
        print("icov shape", icov.shape)
        print("icov contig", icov.flags["C_CONTIGUOUS"])
        fisher = ksw.compute_fisher_isotropic(icov, comm=mpi_comm)
    logger.info("Fisher: %s, standard deviation: %s", fisher, np.sqrt(1 / fisher))

    # note that these are not fully loaded into memory, yet
    alms = data_file["alm_lensed"] if core.lensing else data_file["alm"]
    print("alm dtype:", alms.dtype)

    # def alm_loader(idx):
    #     """load the alms into memory with debug logging"""
    #     logger.debug("Sending alm %s", idx)
    #     return icov_func(core, alms[idx, pol_idxs])

    pol_idxs = core.pol_idxs(pretrimmed=True)
    estimates, _, _, _ = ksw.compute_estimate_batch(
        lambda idx: icov_func(icov_ell, alms[idx, pol_idxs]),
        # lambda idx: alms[idx, pol_idxs],
        range(core.num_estimates),
        comm=mpi_comm,
        fisher=fisher,
        theta_batch=theta_batch,
        lin_term=0 if not core.force_ksw else None,
    )

    if mpi_root:
        logger.info("Saving data")

        fnls = np.array(data_file["fnl"][:])
        print(fnls)
        data_file.close()

        # save the data, this will append to the alm_file
        sdata = {}
        sdata["fisher"] = np.atleast_1d(fisher)
        sdata["estimate"] = estimates
        sdata["error"] = (estimates - fnls) * np.sqrt(fisher)
        save_data(core.file_complete, sdata, mode="a")

        # and lets make some plots
        plot_dir = os.path.join(core.plot_dir, "estimator")
        os.makedirs(plot_dir, exist_ok=True)

        pred_file = os.path.join(plot_dir, f"{core.base_name}_{core.sjob}_preds.png")
        plot_predictions(fnls, estimates, fisher=fisher, save_file=pred_file)

        hist_file = os.path.join(plot_dir, f"{core.base_name}_{core.sjob}_hist.png")
        plot_histogram(fnls, estimates, save_file=hist_file)
        print_errors(fnls, estimates, fisher)

    logger.info("Finished %s!", mpi_rank)


if __name__ == "__main__":
    sys.exit(main())
