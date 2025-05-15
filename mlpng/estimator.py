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


class Estimator(Generator):
    def __init__(self, log_level=None, **kwargs):
        if log_level is None:
            log_level = logging.DEBUG if mpi_root else logging.ERROR

        super().__init__(log_level=log_level, **kwargs)
        self.logger = setup_logging(name=f"mlpng.estimator_{mpi_rank}", level=log_level)

        # the KSW code requires the total_sims to be >= mpi_size
        # best usage would have total_sims % mpi_size == 0, but not required
        assert self.num_estimates >= mpi_size, (
            "total_sims < mpi_size, lower ntasks or increase sims"
        )

        self.logger.info(
            "Computing %s estimates in %.2f batches",
            self.num_estimates,
            self.num_estimates / mpi_size,
        )
        if self.num_estimates % mpi_size != 0:
            self.logger.warning(
                "num_estimates is not divisible by mpi_size, "
                "this will lead to uneven workloads."
            )

    def run(self, lensing=False):
        l_str = "lensed" if lensing else "unlensed"
        theta_batch = int(np.floor(1.5 * self.lmax + 1)) // mpi_size

        for shape in self.shapes:
            self.logger.info(
                "Estimating for %s %s, using %s estimates",
                l_str,
                shape,
                self.num_estimates,
            )
            with h5py.File(self.file, "r", swmr=True, locking=False) as data:
                alm_l = np.array(data["alm_l"][l_str][shape][: self.num_estimates])
                alm_nl = np.array(data["alm_nl"][l_str][shape][: self.num_estimates])
                fnl = self.rng.uniform(
                    self.fnl_min, self.fnl_max, (self.num_estimates, 1, 1)
                )

                # make our alms
                alm = alm_l + fnl * alm_nl

                self.logger.debug("Computing estimates for %s %s", l_str, shape)
                ksw = self.get_ksw(shape)
                fisher = ksw.compute_fisher()
                self.logger.debug("Fisher: %s, std: %s", fisher, 1 / np.sqrt(fisher))

                estimates, _, _, _ = ksw.compute_estimate_batch(
                    lambda idx: self.icov_func(alm[idx]),
                    range(self.num_estimates),
                    comm=mpi_comm,
                    fisher=fisher,
                    theta_batch=theta_batch,
                    lin_term=None,
                )

            mpi_comm.Barrier()
            if mpi_root:
                # save the data, this will append to the alm_file
                sdata = {"estimates": {l_str: {shape: estimates}}}
                save_data(self.file, sdata, mode="a", verbose=True)

                fnl = fnl.flatten()
                print_errors(fnl, estimates, fisher)
                if self.plot:
                    base = f"ksw_{shape}_{l_str}"
                    plot_predictions(
                        fnl,
                        estimates,
                        fisher=fisher,  # type: ignore
                        save_file=self.get_plot_file(f"{base}_preds"),
                    )
                    plot_histogram(
                        fnl,
                        estimates,
                        save_file=self.get_plot_file(f"{base}_hist"),
                    )

            self.logger.debug("Finished %s", shape)

        self.logger.info("Finished estimating %s", l_str)


if __name__ == "__main__":
    estimator = Estimator()

    estimator.run(False)

    if estimator.lensing:
        estimator.run(True)
