import os
import logging
import numpy as np
from joblib import Parallel, delayed

from ksw import KSW as OriginalKSW, utils, estimator_core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def process_batch_step(
    tidx_start,
    theta_batch,
    thetas,
    theta_weights,
    lmax,
    dtype,
    rule,
    weights,
    f_i_ell,
    a_ell_m,
    grad_t,
    nphi,
):
    # side affects, needed and read only
    grad_t = grad_t.copy()

    thetas_batch = thetas[tidx_start : tidx_start + theta_batch]
    ct_weights_batch = theta_weights[tidx_start : tidx_start + theta_batch]
    y_m_ell = estimator_core.compute_ylm(thetas_batch, lmax, dtype=dtype)
    estimator_core.step(
        ct_weights_batch,
        rule,
        weights,
        f_i_ell.copy(),
        a_ell_m.copy(),
        y_m_ell,
        grad_t,
        nphi,
    )
    return grad_t


def process_batch_estimate(
    tidx_start,
    theta_batch,
    thetas,
    theta_weights,
    lmax,
    dtype,
    rule,
    weights,
    f_i_ell,
    a_ell_m,
    nphi,
):
    thetas_batch = thetas[tidx_start : tidx_start + theta_batch]
    ct_weights_batch = theta_weights[tidx_start : tidx_start + theta_batch]
    y_m_ell = estimator_core.compute_ylm(thetas_batch, lmax, dtype=dtype)
    return estimator_core.compute_estimate(
        ct_weights_batch, rule, weights, f_i_ell.copy(), a_ell_m.copy(), y_m_ell, nphi
    )


class KSW_joblib(OriginalKSW):
    """
    Jupyter on the clusters doesn't like MPI, so this uses joblib to parallelize the loop.
    I made some small changes to the logging, so that it is clear that this is being used,
    and some minor copy have been added due to pickling

    Example Import:
    ```python
    from scripts import core # note the small c to import the core module and not the class
    from ksw_joblib import KSW_joblib

    # Monkey patch the Core module
    core.KSW = KSW_joblib
    ```

    Also See:
        the simulator.ipynb notebook in the same directory.
    """

    def __init__(self, *args, **kwargs):
        logger.info("Using KSW_joblib")
        super().__init__(*args, **kwargs)

        temp_folder = os.environ.get("SCRATCH", None)
        self.parallel = Parallel(
            n_jobs=-1, return_as="generator", temp_folder=temp_folder
        )

    def _step(self, alm, theta_batch=25):
        alm = utils.alm_return_2d(alm, self.npol, self.lmax)
        alm = self.icov(alm)

        a_ell_m = utils.alm2a_ell_m(alm)
        a_ell_m = a_ell_m.astype(self.cdtype)
        grad_t = np.zeros_like(a_ell_m)

        red_bisp = self.red_bispectra[0]
        f_i_ell, rule, weights = self._init_reduced_bispectrum(red_bisp)

        # Use joblib to parallelize the loop
        results = self.parallel(
            delayed(process_batch_step)(
                tidx_start,
                theta_batch,
                self.thetas,
                self.theta_weights,
                self.lmax,
                self.dtype,
                rule,
                weights,
                f_i_ell,
                a_ell_m,
                grad_t,
                self.nphi,
            )
            for tidx_start in range(0, len(self.thetas), theta_batch)
        )
        grad_t = sum(results)

        # Turn back into healpy shape.
        grad_t = utils.a_ell_m2alm(grad_t).astype(self.cdtype)
        return grad_t

    def _process_file_step(self, alm_loader, alm_file, **kwargs):
        logger.info("Processing %s", alm_file)
        alm = alm_loader(alm_file)
        grad_t = self._step(alm, **kwargs)
        mc_gt_sq = utils.contract_almxblm(grad_t, self.icov(self.beam(np.conj(grad_t))))
        return grad_t, mc_gt_sq

    def step_batch(self, alm_loader, alm_files, **kwargs):
        # Monte carlo quantities local to rank.
        mc_idx_loc = 0
        mc_gt_sq_loc = 0
        mc_gt_loc = 0

        # Combine the results
        for alm_file in alm_files:
            grad_t, mc_gt_sq = self._process_file_step(alm_loader, alm_file, **kwargs)
            mc_gt_loc += grad_t
            mc_gt_sq_loc += mc_gt_sq
            mc_idx_loc += 1

        self.mc_gt = mc_gt_loc
        self.mc_gt_sq = mc_gt_sq_loc
        self.mc_idx = mc_idx_loc

    def compute_estimate_batch(self, alm_loader, alm_files, **kwargs):
        estimates = np.zeros(len(alm_files))
        fisher = kwargs.pop("fisher", self.compute_fisher())

        # Split alm_file loop over ranks.
        for aidx in range(len(alm_files)):
            alm_file = alm_files[aidx]
            alm = alm_loader(alm_file)

            estimate = self.compute_estimate(alm, fisher=fisher, **kwargs)
            logger.info("Estimate: {}".format(estimate))

            estimates[aidx] = estimate

        return estimates

    def compute_estimate(self, alm, theta_batch=25, fisher=None, lin_term=None):
        # Similar to step, but only do backward transform, multiply alm with linear term
        # and apply normalization.

        alm = utils.alm_return_2d(alm, self.npol, self.lmax)
        alm = self.icov(alm)

        t_cubic = 0  # The cubic estimate.
        if fisher is None:
            fisher = self.compute_fisher()
        if lin_term is None:
            lin_term = self.compute_linear_term(alm, no_icov=True)

        a_ell_m = utils.alm2a_ell_m(alm)
        a_ell_m = a_ell_m.astype(self.cdtype)

        red_bisp = self.red_bispectra[0]
        f_i_ell, rule, weights = self._init_reduced_bispectrum(red_bisp)

        estimates = self.parallel(
            delayed(process_batch_estimate)(
                tidx_start,
                theta_batch,
                self.thetas,
                self.theta_weights,
                self.lmax,
                self.dtype,
                rule,
                weights,
                f_i_ell,
                a_ell_m,
                self.nphi,
            )
            for tidx_start in range(0, len(self.thetas), theta_batch)
        )

        t_cubic = sum(estimates)
        logger.debug("t_cubic: %s, lin_term: %s, fisher: %s", t_cubic, lin_term, fisher)
        return (t_cubic - lin_term) / fisher
