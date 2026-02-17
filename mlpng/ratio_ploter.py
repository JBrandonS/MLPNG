"""
Compute R_GL = (S/N)_GL / (S/N)_0 as a function of phi_scale.

Extends the Generator class to compute the flat-sky S/N degradation ratio
due to gravitational lensing, sweeping over phi_scale values. Uses Vegas
Monte Carlo integration for the bispectrum signal, Gaussian noise variance,
and lensing trispectrum correction, following the formalism from
Babich & Zaldarriaga (2004) and Serra & Cooray, as implemented in
notebooks/lens_test.ipynb and notebooks/phi_scale_impact.ipynb.

Usage:
    python -m mlpng.ratio_ploter settings/n256.json --pols T --shapes local
"""

import logging
import os
from math import sqrt

import matplotlib.pyplot as plt
import numba
import numpy as np
import vegas
from scipy.interpolate import interp1d
from ksw.radial_functional import radial_func

from mlpng.generator import Generator
from mlpng.utils import setup_logging, save_data

logger = logging.getLogger(__name__)

# --- phi_scale values to sweep over ---
PHI_SCALES = [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500]


# ============================================================================
# Numba-JIT'd kernels (ported from notebooks/lens_test.ipynb)
# ============================================================================


@numba.njit(nogil=True, fastmath=False)
def numba_trapezoid(y, x):
    n = len(x)
    integral = 0.0
    for i in range(1, n):
        integral += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return integral


@numba.njit(nogil=True, fastmath=False)
def compute_bispectrum_local(alpha, beta, l1, l2, l3, rs):
    integrand = (
        rs
        * rs
        * (
            beta[l3] * beta[l2] * alpha[l1]
            + beta[l1] * beta[l3] * alpha[l2]
            + beta[l2] * beta[l1] * alpha[l3]
        )
    )
    return 6.0 / 5.0 * numba_trapezoid(integrand, x=rs)


@numba.njit(nogil=True, fastmath=False)
def compute_W(l1, l2, l3, c_ell, radii, alpha, beta):
    C6 = 6.0 * c_ell[l1] * c_ell[l2] * c_ell[l3]
    B = compute_bispectrum_local(alpha, beta, l1, l2, l3, radii)
    return 4 * np.pi * np.pi * B / C6


# --- Trispectrum ---


@numba.njit(nogil=True, fastmath=False)
def tri_single(lmax, c_ell, c_ell_phi, l1_0, l1_1, l2_0, l2_1, l3_0, l3_1, l4_0, l4_1):
    l13_0 = l1_0 + l3_0
    l13_1 = l1_1 + l3_1
    l24_0 = l2_0 + l4_0
    l24_1 = l2_1 + l4_1

    l13_mag = int(round(sqrt(l13_0 * l13_0 + l13_1 * l13_1)))
    l3_mag = int(round(sqrt(l3_0 * l3_0 + l3_1 * l3_1)))
    l4_mag = int(round(sqrt(l4_0 * l4_0 + l4_1 * l4_1)))

    if not (2 < l3_mag <= lmax and 2 < l4_mag <= lmax and 2 < l13_mag <= lmax):
        return 0.0

    l13_3 = l13_0 * l3_0 + l13_1 * l3_1
    l24_4 = l24_0 * l4_0 + l24_1 * l4_1
    return c_ell_phi[l13_mag] * c_ell[l3_mag] * c_ell[l4_mag] * l13_3 * l24_4


@numba.njit(nogil=True, fastmath=False)
def tri(lmax, c_ell, c_ell_phi, l1_0, l1_1, l2_0, l2_1, l3_0, l3_1, l4_0, l4_1):
    value = 0.0
    value += tri_single(
        lmax, c_ell, c_ell_phi, l1_0, l1_1, l2_0, l2_1, l3_0, l3_1, l4_0, l4_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l1_0, l1_1, l2_0, l2_1, l4_0, l4_1, l3_0, l3_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l1_0, l1_1, l3_0, l3_1, l2_0, l2_1, l4_0, l4_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l1_0, l1_1, l3_0, l3_1, l4_0, l4_1, l2_0, l2_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l1_0, l1_1, l4_0, l4_1, l2_0, l2_1, l3_0, l3_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l1_0, l1_1, l4_0, l4_1, l3_0, l3_1, l2_0, l2_1
    )

    value += tri_single(
        lmax, c_ell, c_ell_phi, l2_0, l2_1, l1_0, l1_1, l3_0, l3_1, l4_0, l4_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l2_0, l2_1, l1_0, l1_1, l4_0, l4_1, l3_0, l3_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l2_0, l2_1, l3_0, l3_1, l1_0, l1_1, l4_0, l4_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l2_0, l2_1, l3_0, l3_1, l4_0, l4_1, l1_0, l1_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l2_0, l2_1, l4_0, l4_1, l1_0, l1_1, l3_0, l3_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l2_0, l2_1, l4_0, l4_1, l3_0, l3_1, l1_0, l1_1
    )

    value += tri_single(
        lmax, c_ell, c_ell_phi, l3_0, l3_1, l1_0, l1_1, l2_0, l2_1, l4_0, l4_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l3_0, l3_1, l1_0, l1_1, l4_0, l4_1, l2_0, l2_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l3_0, l3_1, l2_0, l2_1, l1_0, l1_1, l4_0, l4_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l3_0, l3_1, l2_0, l2_1, l4_0, l4_1, l1_0, l1_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l3_0, l3_1, l4_0, l4_1, l1_0, l1_1, l2_0, l2_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l3_0, l3_1, l4_0, l4_1, l2_0, l2_1, l1_0, l1_1
    )

    value += tri_single(
        lmax, c_ell, c_ell_phi, l4_0, l4_1, l1_0, l1_1, l2_0, l2_1, l3_0, l3_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l4_0, l4_1, l1_0, l1_1, l3_0, l3_1, l2_0, l2_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l4_0, l4_1, l2_0, l2_1, l1_0, l1_1, l3_0, l3_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l4_0, l4_1, l2_0, l2_1, l3_0, l3_1, l1_0, l1_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l4_0, l4_1, l3_0, l3_1, l1_0, l1_1, l2_0, l2_1
    )
    value += tri_single(
        lmax, c_ell, c_ell_phi, l4_0, l4_1, l3_0, l3_1, l2_0, l2_1, l1_0, l1_1
    )

    return value / 2.0


# ============================================================================
# S integrand (signal)
# ============================================================================


@numba.njit(nogil=True, fastmath=False)
def s_integrand_scalar(x, lmax, c_ell, alpha, beta, radii):
    l1_0, l1_1 = x[0], x[1]
    l2_0, l2_1 = x[2], x[3]
    l3_0, l3_1 = -l1_0 - l2_0, -l1_1 - l2_1

    l1_mag = int(round(sqrt(l1_0 * l1_0 + l1_1 * l1_1)))
    l2_mag = int(round(sqrt(l2_0 * l2_0 + l2_1 * l2_1)))
    l3_mag = int(round(sqrt(l3_0 * l3_0 + l3_1 * l3_1)))

    if not (2 < l1_mag <= lmax and 2 < l2_mag <= lmax and 2 < l3_mag <= lmax):
        return 0.0

    W = compute_W(l1_mag, l2_mag, l3_mag, c_ell, radii, alpha, beta)
    B = compute_bispectrum_local(alpha, beta, l1_mag, l2_mag, l3_mag, radii)
    return W * B


@numba.njit(parallel=True, nogil=True, fastmath=False)
def s_integrand_batch(x_batch, lmax, c_ell, alpha, beta, radii):
    n = x_batch.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        out[i] = s_integrand_scalar(x_batch[i], lmax, c_ell, alpha, beta, radii)
    return out


class SIntegrandNumbaVec(vegas.LBatchIntegrand):
    def __init__(self, lmax, c_ell, alpha, beta, radii):
        self.lmax = lmax
        self.c_ell = c_ell
        self.alpha = alpha
        self.beta = beta
        self.radii = radii

    def __call__(self, x):
        return s_integrand_batch(
            x, self.lmax, self.c_ell, self.alpha, self.beta, self.radii
        )


# ============================================================================
# N^2 integrand (Gaussian noise variance)
# ============================================================================


@numba.njit(nogil=True, fastmath=False)
def n2_integrand_scalar(x, lmax, c_ell, c_ell_lens, alpha, beta, radii):
    l1_0, l1_1 = x[0], x[1]
    l2_0, l2_1 = x[2], x[3]
    l3_0, l3_1 = -l1_0 - l2_0, -l1_1 - l2_1

    lm = lmax
    if not (-lm <= l3_0 <= lm and -lm <= l3_1 <= lm):
        return 0.0

    l1 = int(round(sqrt(l1_0 * l1_0 + l1_1 * l1_1)))
    l2 = int(round(sqrt(l2_0 * l2_0 + l2_1 * l2_1)))
    l3 = int(round(sqrt(l3_0 * l3_0 + l3_1 * l3_1)))

    if not (2 < l1 <= lmax and 2 < l2 <= lmax and 2 < l3 <= lmax):
        return 0.0

    cl = c_ell_lens if c_ell_lens is not None else c_ell
    C6 = 6.0 * cl[l1] * cl[l2] * cl[l3]
    W = compute_W(l1, l2, l3, c_ell, radii, alpha, beta)

    return W * W * C6


@numba.njit(parallel=True, nogil=True, fastmath=False)
def n2_integrand_batch(x_batch, lmax, c_ell, c_ell_lens, alpha, beta, radii):
    n = x_batch.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        out[i] = n2_integrand_scalar(
            x_batch[i], lmax, c_ell, c_ell_lens, alpha, beta, radii
        )
    return out


class N2IntegrandNumbaVec(vegas.LBatchIntegrand):
    def __init__(self, lmax, c_ell, c_ell_lens, alpha, beta, radii):
        self.lmax = lmax
        self.c_ell = c_ell
        self.c_ell_lens = c_ell_lens
        self.alpha = alpha
        self.beta = beta
        self.radii = radii

    def __call__(self, x):
        return n2_integrand_batch(
            x, self.lmax, self.c_ell, self.c_ell_lens, self.alpha, self.beta, self.radii
        )


# ============================================================================
# delta N^2 integrand (lensing trispectrum correction)
# ============================================================================


@numba.njit(nogil=True, fastmath=False)
def delta_n2_integrand_scalar(x, lmax, c_ell, c_ell_lens, cl_phi, alpha, beta, radii):
    l1_0, l1_1 = x[0], x[1]
    l2_0, l2_1 = x[2], x[3]
    lp2_0, lp2_1 = x[4], x[5]

    l3_0, l3_1 = -l1_0 - l2_0, -l1_1 - l2_1
    lp1_0, lp1_1 = -l1_0, -l1_1
    lp3_0, lp3_1 = -l2_0 - l3_0 - lp2_0, -l2_1 - l3_1 - lp2_1

    lm = lmax
    if not (
        -lm <= l3_0 <= lm
        and -lm <= l3_1 <= lm
        and -lm <= lp1_0 <= lm
        and -lm <= lp1_1 <= lm
        and -lm <= lp3_0 <= lm
        and -lm <= lp3_1 <= lm
    ):
        return 0.0

    l1 = int(round(sqrt(l1_0 * l1_0 + l1_1 * l1_1)))
    l2 = int(round(sqrt(l2_0 * l2_0 + l2_1 * l2_1)))
    l3 = int(round(sqrt(l3_0 * l3_0 + l3_1 * l3_1)))
    lp1 = int(round(sqrt(lp1_0 * lp1_0 + lp1_1 * lp1_1)))
    lp2 = int(round(sqrt(lp2_0 * lp2_0 + lp2_1 * lp2_1)))
    lp3 = int(round(sqrt(lp3_0 * lp3_0 + lp3_1 * lp3_1)))

    if not (
        2 < l1 <= lmax
        and 2 < l2 <= lmax
        and 2 < l3 <= lmax
        and 2 < lp1 <= lmax
        and 2 < lp2 <= lmax
        and 2 < lp3 <= lmax
    ):
        return 0.0

    W = compute_W(l1, l2, l3, c_ell, radii, alpha, beta)
    Wp = compute_W(lp1, lp2, lp3, c_ell, radii, alpha, beta)
    T = tri(lmax, c_ell, cl_phi, l2_0, l2_1, l3_0, l3_1, lp2_0, lp2_1, lp3_0, lp3_1)
    return W * Wp * c_ell_lens[l1] * T


@numba.njit(parallel=True, nogil=True, fastmath=False)
def delta_n2_integrand_batch(x_batch, lmax, c_ell, c_ell_lens, cl_phi, alpha, beta, radii):
    n = x_batch.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        out[i] = delta_n2_integrand_scalar(
            x_batch[i], lmax, c_ell, c_ell_lens, cl_phi, alpha, beta, radii
        )
    return out


class DeltaN2IntegrandNumbaVec(vegas.LBatchIntegrand):
    def __init__(self, lmax, c_ell, c_ell_lens, cl_phi, alpha, beta, radii):
        self.lmax = lmax
        self.c_ell = c_ell
        self.c_ell_lens = c_ell_lens
        self.cl_phi = cl_phi
        self.alpha = alpha
        self.beta = beta
        self.radii = radii

    def __call__(self, x):
        return delta_n2_integrand_batch(
            x,
            self.lmax,
            self.c_ell,
            self.c_ell_lens,
            self.cl_phi,
            self.alpha,
            self.beta,
            self.radii,
        )


# ============================================================================
# Integration driver functions
# ============================================================================


def compute_S(c_ell, alpha_l, beta_l, radii, lmax, neval=1e6, nitn=(10, 10)):
    """Compute the signal S via 4D Vegas MC integration."""
    integrand = SIntegrandNumbaVec(lmax, c_ell, alpha_l.T, beta_l.T, radii)
    integ = vegas.Integrator([[-lmax, lmax]] * 4)
    integ(integrand, nitn=nitn[0], neval=neval)
    result = integ(integrand, nitn=nitn[1], neval=neval)
    return result.mean / (2 * np.pi) ** 4 / np.pi


def compute_N2(c_ell, alpha_l, beta_l, radii, lmax, c_ell_lens=None, neval=1e6, nitn=(10, 10), integ=None):
    """Compute <N^2> (Gaussian noise variance) via 4D Vegas MC integration.

    Parameters:
        c_ell: Unlensed C_ell (always used for W weights).
        alpha_l, beta_l: Transfer function arrays, shape (n_radii, nell).
        radii: Radial integration grid.
        lmax: Maximum multipole.
        c_ell_lens: If provided, used for the 6-point function C_6. Otherwise uses c_ell.
        neval: Number of Vegas evaluations per iteration.
        nitn: Tuple of (warmup, production) iteration counts.
        integ: Optional pre-adapted vegas.Integrator to reuse.

    Returns:
        (value, integrator) tuple.
    """
    integrand = N2IntegrandNumbaVec(lmax, c_ell, c_ell_lens, alpha_l.T, beta_l.T, radii)
    if integ is None:
        integ = vegas.Integrator([[-lmax, lmax]] * 4)
        integ(integrand, nitn=nitn[0], neval=neval)
    result = integ(integrand, nitn=nitn[1], neval=neval)
    return result.mean / (2 * np.pi) ** 6 / np.pi, integ


def compute_delta_N2(c_ell, c_ell_lens, cl_phi, alpha_l, beta_l, radii, lmax, neval=1e6, nitn=(10, 10)):
    """Compute delta<N^2> (connected 4-point / trispectrum correction) via 6D Vegas MC integration.

    Parameters:
        c_ell: Unlensed C_ell (used for W weights and trispectrum).
        c_ell_lens: Lensed C_ell (CAMB output, used for the external leg).
        cl_phi: Lensing potential C_ell^{phi phi}, already scaled by phi_scale^2.
        alpha_l, beta_l: Transfer function arrays, shape (n_radii, nell).
        radii: Radial integration grid.
        lmax: Maximum multipole.
        neval: Number of Vegas evaluations per iteration.
        nitn: Tuple of (warmup, production) iteration counts.

    Returns:
        Scalar delta<N^2> value.
    """
    integrand = DeltaN2IntegrandNumbaVec(
        lmax, c_ell, c_ell_lens, cl_phi, alpha_l.T, beta_l.T, radii
    )
    integ = vegas.Integrator([[-lmax, lmax]] * 6, neval=int(neval))
    integ(integrand, nitn=nitn[0], neval=int(neval))
    result = integ(integrand, nitn=nitn[1], neval=int(neval))
    return 9.0 / (2 * np.pi) ** 8 * result.mean / np.pi


# ============================================================================
# RatioPloter class
# ============================================================================


class RatioPloter(Generator):
    """Compute R_GL = (S/N)_GL / (S/N)_0 vs phi_scale.

    Extends Generator to access cosmology, transfer functions, and power spectra.
    Then uses flat-sky Vegas MC integration (Babich & Zaldarriaga formalism) to
    compute the S/N degradation ratio as a function of the lensing potential
    scale factor phi_scale.
    """

    def __init__(self, argv=None, log_level=logging.DEBUG):
        super().__init__(argv, log_level=log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def _compute_alpha_beta(self):
        """Compute alpha_ell(r) and beta_ell(r) transfer functions.

        Uses the same physics as generator.py / lens_test.ipynb:
        CAMB transfer functions, zeta->phi conversion (5/3 factor),
        primordial power spectrum with (3/5)^2 factor, radial_func
        from KSW, and cubic interpolation to the working ell grid.

        Returns:
            alpha_l: array of shape (n_radii, nell)
            beta_l: array of shape (n_radii, nell)
        """
        pol_idxs = self.pol_idxs()
        transfer = self.cosmo.transfer

        tr_ells = transfer["ells"]
        tr_k = transfer["k"]
        tr_ell_k = transfer["tr_ell_k"][..., pol_idxs]
        tr_ell_k = tr_ell_k * (5 / 3)  # zeta -> phi conversion

        f_k = np.ones((len(tr_k), 2), dtype=self.r_dtype)

        # Primordial power spectrum -> delta_phi
        Pk = self.cosmo.camb_params.primordial_power(tr_k, 0)
        delta_phi = 2 * np.pi**2 * Pk * (3 / 5) ** 2

        f_k[:, 1] = delta_phi * tr_k ** (-3)  # beta weighting

        # Radial integration via KSW's radial_func
        rad = radial_func(f_k, tr_ell_k, tr_k, self.radii, tr_ells)

        # Interpolate from sparse tr_ells to full ell range
        alpha_ell = rad[..., 0, 0]
        alpha_l = interp1d(
            tr_ells, alpha_ell, kind="cubic", axis=1,
            bounds_error=False, fill_value=0,
        )(self.ells)

        beta_ell = rad[..., 0, 1]
        beta_l = interp1d(
            tr_ells, beta_ell, kind="cubic", axis=1,
            bounds_error=False, fill_value=0,
        )(self.ells)

        return alpha_l, beta_l

    def run(self, neval=1e6, nitn=(10, 10), verbose=False):
        """Run the R_GL vs phi_scale computation.

        For each phi_scale value:
        1. Scale cl_phi by phi_scale^2 (following phi_scale_impact.ipynb convention)
        2. Compute delta<N^2> (only quantity that depends on phi_scale)
        3. Compute R_GL = sqrt(N2_0 / (N2_lens + delta_N2))

        Alpha/beta, N2_0, and N2_lens are computed once since they don't depend on phi_scale.
        """
        self.logger.info("Computing alpha and beta transfer functions...")
        alpha_l, beta_l = self._compute_alpha_beta()

        c_ell = self.c_ell[0]  # TT only
        c_ell_lens = self.c_ell_lens[0]
        cl_phi_base = self.cl_phi  # unscaled C_ell^{phi phi}
        radii = self.radii
        lmax = self.lmax

        self.logger.info("Computing N2_0 (unlensed)...")
        N2_0, integ = compute_N2(c_ell, alpha_l, beta_l, radii, lmax, neval=neval, nitn=nitn)
        self.logger.info("N2_0 = %e", N2_0)

        self.logger.info("Computing N2_lens (lensed, phi_scale-independent)...")
        N2_lens, integ = compute_N2(
            c_ell, alpha_l, beta_l, radii, lmax,
            c_ell_lens=c_ell_lens, neval=neval, nitn=nitn, integ=integ,
        )
        self.logger.info("N2_lens = %e", N2_lens)

        R_gl_values = []
        delta_N2_values = []

        for phi_scale in PHI_SCALES:
            self.logger.info("--- phi_scale = %s ---", phi_scale)

            # Scale the lensing potential power spectrum by phi_scale^2
            # (following phi_scale_impact.ipynb convention)
            cl_phi_scaled = cl_phi_base * (phi_scale ** 2)

            self.logger.info("Computing delta_N2 for phi_scale=%s...", phi_scale)
            delta_N2 = compute_delta_N2(
                c_ell, c_ell_lens, cl_phi_scaled,
                alpha_l, beta_l, radii, lmax,
                neval=neval, nitn=nitn,
            )
            delta_N2_values.append(delta_N2)

            R_gl = sqrt(N2_0 / (N2_lens + delta_N2))
            R_gl_values.append(R_gl)

            self.logger.info(
                "phi_scale=%s: delta_N2=%e, R_GL=%.4f", phi_scale, delta_N2, R_gl
            )

        # Convert to arrays for saving
        phi_scales_arr = np.array(PHI_SCALES, dtype=self.r_dtype)
        R_gl_arr = np.array(R_gl_values, dtype=self.r_dtype)
        delta_N2_arr = np.array(delta_N2_values, dtype=self.r_dtype)

        # Save data to HDF5
        sdata = {
            "ratio_plot": {
                "phi_scales": phi_scales_arr,
                "R_gl": R_gl_arr,
                "N2_0": np.atleast_1d(np.array(N2_0, dtype=self.r_dtype)),
                "N2_lens": np.atleast_1d(np.array(N2_lens, dtype=self.r_dtype)),
                "delta_N2": delta_N2_arr,
                "lmax": np.atleast_1d(np.array(lmax, dtype=np.int32)),
            }
        }
        save_data(self.file, sdata, verbose=verbose)
        self.logger.info("Data saved to %s", self.file)

        # Generate plot
        if self.should_plot():
            self._make_plot(phi_scales_arr, R_gl_arr)

        self.logger.info("Done!")

    def _make_plot(self, phi_scales, R_gl):
        """Generate and save the R_GL vs phi_scale plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(phi_scales, R_gl, "o-", color="tab:blue", linewidth=2, markersize=6, label=r"$R_{GL}$")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label=r"$R=1$ (no degradation)")

        ax.set_xlabel(r"$\phi_{\mathrm{scale}}$", fontsize=14)
        ax.set_ylabel(r"$R_{GL} = (S/N)_{GL} / (S/N)_0$", fontsize=14)
        ax.set_title(
            rf"S/N Degradation vs Lensing Scale ($\ell_{{\max}}={self.lmax}$, nside={self.nside})",
            fontsize=14,
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()

        plot_dir = os.path.join(
            self.dirs["plot"], self.name, str(self.slurm.job), "ratio_plot"
        )
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = self.get_plot_file("R_gl_vs_phi_scale", plot_dir)
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Plot saved to %s", plot_file)


if __name__ == "__main__":
    setup_logging(
        __name__,
        level=logging.DEBUG,
        scripts_level=logging.DEBUG,
        base_level=logging.ERROR,
    )

    ratio_ploter = RatioPloter()

    # check if we actually need to run / clean files
    ratio_ploter.check_existing_data_file()

    ratio_ploter.run()
