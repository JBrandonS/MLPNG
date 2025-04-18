import logging
import numpy as np
from mlpng.utils import (
    setup_logging,
)
from mlpng.generator import Generator
from ksw import Shape, ReducedBispectrum
from ksw.radial_functional import radial_func
from mpi4py import MPI

logger = setup_logging(__name__, level=logging.DEBUG)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

generator = Generator(["settings/n32.json", "--nsims", "3", "--pols", "T"])
estimator = generator.get_ksw("equilateral", step=False)

shapes = ["local", "equilateral", "orthogonal"]
fishers = generator.compute_fisher_shapes(shapes)

logger.info("%s, %s", fishers.shape, fishers)
logger.info("here")
logger.info("errors: %s", np.sqrt(1 / np.diag(fishers)))
margs = np.sqrt(np.diag(np.linalg.inv(fishers)))
logger.info("Marginal likelihoods: %s", margs)
