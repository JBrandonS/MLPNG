import sys
import logging
import os
import healpy as hp

import h5py
import numpy as np
from mpi4py import MPI

from .generator import Generator
from .utils import save_data, setup_logging, print_errors
from .utils.plots import plot_histogram, plot_predictions

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

# estimator will run multiple jobs per id which get sent to the same log file,
# so we only want to log the root to keep from spamming the log
if mpi_root:
    logger = setup_logging(name=f"mlpng.estimator_{mpi_rank}", level=logging.DEBUG)
else:
    logger = setup_logging(name=f"mlpng.estimator_{mpi_rank}", level=logging.ERROR)

class Estimator(Generator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the KSW code requires the total_sims to be >= mpi_size
        # best usage would have total_sims % mpi_size == 0, but not required
        assert (
            self.num_estimates >= mpi_size
        ), "total_sims < mpi_size, lower ntasks or increase sims"

        logger.info(
            "Computing %s estimates in %.2f batches",
            self.num_estimates,
            self.num_estimates / mpi_size,
        )
        if self.num_estimates % mpi_size != 0:
            logger.warning(
                "num_estimates is not divisible by mpi_size, "
                "this will lead to uneven workloads."
            )

        # The default theta_batch size is 25, which is really small, we want to increase it
        self.theta_batch = int(np.floor(1.5 * self.lmax + 1)) // mpi_size
        logger.debug("Using theta_batch %s", self.theta_batch)

        # if self.isotropic:
        # with h5py.File(self.file, "r", swmr=True, locking=False) as data:
        #     self.fisher = data["fisher_iso"][0]
        # else:
        #     logger.debug("Computing fisher")
        #     if os.path.exists(self.mc_file):
        #         logger.info("Loading KSW state from %s", self.mc_file)
        #         self.ksw.start_from_read_state(self.mc_file, comm=mpi_comm)
        #     else:
        #         alm_steps = self.generate_alm(nsims=self.mc_steps)

        #         logger.info("Initializing KSW with %s steps", self.mc_steps)
        #         self.ksw.step_batch(
        #             lambda i: self.icov_func(alm_steps[i, self.pol_idxs()]),
        #             range(self.mc_steps),
        #             mpi_comm,
        #             theta_batch=self.theta_batch,
        #         )

        #         # save the mc state if we are using the mc file
        #         if self.slurm.is_main:
        #             logger.info("Saving KSW state to %s", self.mc_file)
        #             self.ksw.write_state(self.mc_file, comm=mpi_comm)

        #     self.fisher = self.ksw.compute_fisher()
        # logger.info("Fisher: %s, standard deviation: %s", self.fisher, 1 / np.sqrt(self.fisher))

    def run(self, lensing=False):
        lstr = "lensed" if lensing else "unlensed"
        for shape in self.shapes:
            logger.debug("Estimating for %s %s", lstr, shape)
            with h5py.File(self.file, "r", swmr=True, locking=False) as data:
                alm_l = data["alm_l"][: self.num_estimates, self.pol_idxs()]
                alm_nl = data[lstr][shape]["alm_nl"][: self.num_estimates]
                fnl = np.array(data["fnl"][: self.num_estimates])
                fisher = data[lstr][shape]["fisher"][0]  # TODO fix

                logger.debug(
                    "Fisher: %s, standard deviation: %s", fisher, 1 / np.sqrt(fisher)
                )

                # make our alms
                alm = alm_l + fnl[..., None] * alm_nl

                logger.debug("Computing estimates")
                ksw = self.get_ksw(shape)
                estimates, _, _, _ = ksw.compute_estimate_batch(
                    lambda idx: self.icov_func(alm[idx]),
                    range(self.num_estimates),
                    comm=mpi_comm,
                    fisher=fisher,
                    theta_batch=self.theta_batch,
                    lin_term=0 if self.isotropic else None,
                )

            if mpi_root:
                # save the data, this will append to the alm_file
                sdata = {lstr: {shape: {"estimate": estimates}}}
                save_data(self.file, sdata, mode="a")

                print_errors(fnl, estimates, fisher)
                if self.plot:
                    base = f"ksw_{shape}_{lstr}"
                    plot_predictions(
                        fnl,
                        estimates,
                        fisher=fisher,
                        save_file=self.get_plot_file(f"{base}_preds"),
                    )
                    plot_histogram(
                        fnl, estimates, save_file=self.get_plot_file(f"{base}_hist")
                    )

if __name__ == "__main__":
    estimator = Estimator()
    estimator.run(False)

    # if estimator.lensing:
    #     estimator.run(True)
