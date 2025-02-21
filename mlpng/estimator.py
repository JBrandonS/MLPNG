import logging
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
logger.debug("MPI rank %s of %s, is root: %s", mpi_rank, mpi_size, mpi_root)

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

    def run(self, lensing=False):
        l_str = "lensed" if lensing else "unlensed"

        for shape in self.shapes:
            logger.info("Estimating for %s %s", l_str, shape)
            with h5py.File(self.file, "r", swmr=True, locking=False) as data:
                alm_l = np.array(data["alm_l"][: self.num_estimates])  # type: ignore
                alm_nl = np.array(data[l_str][shape]["alm_nl"][: self.num_estimates])  # type: ignore
                fnl = np.array(data["fnl"][: self.num_estimates])  # type: ignore

                # TODO fix the need to get only the first,
                # maybe we do need to process all maps into one fisher?
                fisher = np.array(data[l_str][shape]["fisher"])[0]  # type: ignore
                logger.debug("Fisher: %s, std: %s", fisher, 1 / np.sqrt(fisher))

                # make our alms
                alm = alm_l[:, None] + fnl[..., None] * alm_nl[:, None]  # type: ignore

                logger.debug("Computing estimates")
                estimates, _, _, _ = self.get_ksw(shape).compute_estimate_batch(
                    lambda idx: self.icov_func(alm[idx, 0]),
                    range(self.num_estimates),
                    comm=mpi_comm,
                    fisher=fisher,
                    theta_batch=self.theta_batch,
                    lin_term=0 if self.isotropic else None,
                )

            if mpi_root:
                # save the data, this will append to the alm_file
                sdata = {l_str: {shape: {"estimate": estimates}}}
                save_data(self.file, sdata, mode="a")

                print_errors(fnl[:, 0], estimates, fisher)
                if self.plot:
                    base = f"ksw_{shape}_{l_str}"
                    plot_predictions(
                        fnl[:, 0],
                        estimates,
                        fisher=fisher,  # type: ignore
                        save_file=self.get_plot_file(f"{base}_preds"),
                    )
                    plot_histogram(
                        fnl[:, 0],
                        estimates,
                        save_file=self.get_plot_file(f"{base}_hist"),
                    )

if __name__ == "__main__":
    estimator = Estimator()
    estimator.run(False)

    if estimator.lensing:
        estimator.run(True)
