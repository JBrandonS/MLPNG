import logging

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

from mlpng import Core, get_itotcov_ell
from mlpng.utils import (
    setup_logging,
    plot_cl_alm,
    plot_predictions,
    plot_histogram,
    plot_elsner_comp,
)
from mlpng.generator import generate_alm, generate_alm_nl, lens_alms
from mlpng.utils.utils import print_errors
from ksw import Shape, ReducedBispectrum
from ksw.radial_functional import radial_func
from mpi4py import MPI

logger = setup_logging(__name__, level=logging.DEBUG)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

core = Core(
    [
        "settings/n64.json",
        "--nsims",
        "3",
        "--pols",
        "T",
    ]
)


core.init_estimator(verbose=False)

# fisher = core.estimator.compute_fisher_isotropic(core.icov, comm=mpi_comm)
# print(f"Fisher: {fisher}, standard deviation: {1 / np.sqrt(fisher)}")

tr_ell_k = core.cosmo.transfer["tr_ell_k"]
k = core.cosmo.transfer["k"]
ells_sparse = core.cosmo.transfer["ells"]
ns = core.cosmo_params["ns"]
ps = core.cosmo_params["pivot_scalar"]

shapes = []
shapes.append(Shape.prim_local(ns, ps))
shapes.append(Shape.prim_equilateral(ns, ps))
shapes.append(Shape.prim_orthogonal(ns, ps))

red_bispectra = []
amp_factor = 2 * (2 * np.pi**2 * core.cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
for prim_shape in shapes:
    f_k = prim_shape.get_f_k(k)
    amps = np.asarray(prim_shape.amps)
    amps *= amp_factor

    # Call C code.
    red_bisp = radial_func(f_k, tr_ell_k, k, core.radii, ells_sparse)

    factors, rule, weights = core.cosmo._parse_prim_reduced_bispec(
        red_bisp, core.radii, prim_shape.rule, amps
    )

    red_bispectra.append(
        ReducedBispectrum(factors, rule, weights, ells_sparse, prim_shape.name)
    )

logger.info("starting")
fishers = core.estimator.compute_fisher_multi(core.icov, red_bispectra, comm=mpi_comm)
logger.info("%s, %s", fishers.shape, fishers)
logger.info("errors: %s", np.sqrt(1 / np.diag(fishers)))
margs = np.sqrt(np.diag(np.linalg.inv(fishers)))
logger.info("Marginal likelihoods: %s", margs)
