# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: language_level=3
"""
Code largely taken from https://github.com/albakalaja/CMB_fisher_bispectrum/blob/main/scripts/compute_fisher.pyx
Thank you: Alba Kalaja

Converted to a class-based implementation for better reuse.
Extended with trispectrum calculation functionality.
"""

import math
import numpy as np
from scipy.integrate import trapezoid
import itertools as it
import vegas
# import numba

cimport cython
cimport numpy as cnp

cdef extern from "math.h":
    double sqrt(double x)
    double round(double x)
    double abs(double x)

cdef inline double dot(double[::1] a, double[::1] b):
    return a[0]*b[0] + a[1]*b[1]

cdef inline int mag(double[::1] a, double[::1] b):
    return int(round(sqrt(dot(a, b))))

# @numba.njit
cpdef double compute_bispectrum_local(
        double[:, ::1] alpha,
        double[:, ::1] beta,
        int l1, int l2, int l3,
        double[::1] rs
    ):
    cdef:
        int i, n = rs.shape[0]
        double[::1] integrand = np.empty(n, dtype=np.float64)
        double r_sq, beta_prod1, beta_prod2, beta_prod3
        double factor = 6.0 / 5.0

    for i in range(n):
        r_sq = rs[i] * rs[i]
        beta_prod1 = beta[l3, i] * beta[l2, i] * alpha[l1, i]
        beta_prod2 = beta[l1, i] * beta[l3, i] * alpha[l2, i]
        beta_prod3 = beta[l2, i] * beta[l1, i] * alpha[l3, i]
        integrand[i] = r_sq * (beta_prod1 + beta_prod2 + beta_prod3)

    return factor * trapezoid(integrand, x=rs)

def compute_bispectrum_equil(alpha, beta, gamma, delta, l1, l2, l3, r_values):
    return 18.0/5.0*trapezoid(np.power(r_values,2.0)*
                                    (  -beta[l3]*beta[l2]*alpha[l1]-
                                        beta[l1]*beta[l3]*alpha[l2]-
                                        beta[l2]*beta[l1]*alpha[l3]-
                                        2.0*delta[l1]*delta[l2]*delta[l3]+
                                        beta[l1]*gamma[l2]*delta[l3]+
                                        beta[l1]*gamma[l3]*delta[l2]+
                                        beta[l2]*gamma[l1]*delta[l3]+
                                        beta[l2]*gamma[l3]*delta[l1]+
                                        beta[l3]*gamma[l1]*delta[l2]+
                                        beta[l3]*gamma[l2]*delta[l1]),
                                            x = r_values)

def compute_bispectrum_ortho(alpha, beta, gamma, delta, l1, l2, l3, r_values):
    return 18.0/5.0*trapezoid(np.power(r_values,2.0)*
                                    (  -3.0*beta[l3]*beta[l2]*alpha[l1]-
                                        3.0*beta[l1]*beta[l3]*alpha[l2]-
                                        3.0*beta[l2]*beta[l1]*alpha[l3]+
                                        8.0*delta[l1]*delta[l2]*delta[l3]-
                                        3.0*beta[l1]*gamma[l2]*delta[l3]-
                                        3.0*beta[l1]*gamma[l3]*delta[l2]-
                                        3.0*beta[l2]*gamma[l1]*delta[l3]-
                                        3.0*beta[l2]*gamma[l3]*delta[l1]-
                                        3.0*beta[l3]*gamma[l1]*delta[l2]-
                                        3.0*beta[l3]*gamma[l2]*delta[l1]),
                                            x = r_values)


def compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l2, l3, l4):
    # lmax = c_ell_phi.shape[0] - 1

    l13 = l1 + l3
    l14 = l1 + l4

    l1_mag = mag(l1, l1)
    l2_mag = mag(l2, l2)
    l13_mag = mag(l13, l13)
    l14_mag = mag(l14, l14)

    # flat sky approx -> C_l^\delta\delta / (|l|^2) = c_l^\phi
    T =  np.dot(l13, l1) * np.dot(l13, l2) * c_ell_phi[l13_mag]
    T += np.dot(l14, l1) * np.dot(l14, l2) * c_ell_phi[l14_mag]
    return c_ell_TT[l1_mag] * c_ell_TT[l2_mag] * T

def compute_trispectrum(c_ell_TT, c_ell_phi, l1, l2, l3, l4):
    r"""
    T = C_l1^TT C_l2^TT [ ((l1+l3)dot l1 (l1+l3) dot l2) / (|l1+l3|^2)) C_l13^\delta\delta + ((l1+l4) dot l1 (l1+l4) dot l2) / (|l1+l4|^2)) C_l14^\delta\delta ] + permutations
    """
    trispectrum = (
          compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l2, l3, l4)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l3, l2, l4)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l4, l2, l3)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l2, l3, l1, l4)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l2, l4, l1, l3)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l3, l4, l1, l2)
    )

    return trispectrum

cdef double _tri_single(
    float[::1] c_ell_TT,
    float[::1] c_ell_phi,
    double[::1] l1,
    double[::1] l2,
    double[::1] l3,
    double[::1] l4
    ):
    cdef:
        int i, n = l1.shape[0]
        double[2] l13
        double[2] l24

    for i in range(n):
        l13[i] = l1[i] + l3[i]
        l24[i] = l2[i] + l4[i]

    cdef:
        int l13_mag = mag(l13, l13)
        int l3_mag = mag(l3, l3)
        int l4_mag = mag(l4, l4)
        double l13_3 = dot(l13, l3)
        double l24_4 = dot(l24, l4)

    return c_ell_phi[l13_mag] * c_ell_TT[l3_mag] * c_ell_TT[l4_mag] * l13_3 * l24_4

def tri(c_ell_TT, c_ell_phi, l1, l2, l3, l4, lmax=None):
    if lmax is None:
        lmax = np.shape(c_ell_TT)[0] - 1

    # Bounds/validity checks similar to notebook code
    for i in [l1, l2, l3, l4]:
        for j in [l1, l2, l3, l4]:
            if np.array_equal(i, j):
                if not 2 <= mag(i, i) <= lmax:
                    return 0.0
            elif not 2 <= mag(i + j, i + j) <= lmax * sqrt(2):
                return 0.0

    value = 0.0

    value += _tri_single(c_ell_TT, c_ell_phi, l1, l2, l3, l4)
    value += _tri_single(c_ell_TT, c_ell_phi, l1, l2, l4, l3)
    value += _tri_single(c_ell_TT, c_ell_phi, l1, l3, l2, l4)
    value += _tri_single(c_ell_TT, c_ell_phi, l1, l3, l4, l2)
    value += _tri_single(c_ell_TT, c_ell_phi, l1, l4, l2, l3)
    value += _tri_single(c_ell_TT, c_ell_phi, l1, l4, l3, l2)

    value += _tri_single(c_ell_TT, c_ell_phi, l2, l1, l3, l4)
    value += _tri_single(c_ell_TT, c_ell_phi, l2, l1, l4, l3)
    value += _tri_single(c_ell_TT, c_ell_phi, l2, l3, l1, l4)
    value += _tri_single(c_ell_TT, c_ell_phi, l2, l3, l4, l1)
    value += _tri_single(c_ell_TT, c_ell_phi, l2, l4, l1, l3)
    value += _tri_single(c_ell_TT, c_ell_phi, l2, l4, l3, l1)

    value += _tri_single(c_ell_TT, c_ell_phi, l3, l1, l2, l4)
    value += _tri_single(c_ell_TT, c_ell_phi, l3, l1, l4, l2)
    value += _tri_single(c_ell_TT, c_ell_phi, l3, l2, l1, l4)
    value += _tri_single(c_ell_TT, c_ell_phi, l3, l2, l4, l1)
    value += _tri_single(c_ell_TT, c_ell_phi, l3, l4, l1, l2)
    value += _tri_single(c_ell_TT, c_ell_phi, l3, l4, l2, l1)

    value += _tri_single(c_ell_TT, c_ell_phi, l4, l1, l2, l3)
    value += _tri_single(c_ell_TT, c_ell_phi, l4, l1, l3, l2)
    value += _tri_single(c_ell_TT, c_ell_phi, l4, l2, l1, l3)
    value += _tri_single(c_ell_TT, c_ell_phi, l4, l2, l3, l1)
    value += _tri_single(c_ell_TT, c_ell_phi, l4, l3, l1, l2)
    value += _tri_single(c_ell_TT, c_ell_phi, l4, l3, l2, l1)

    return value / 2.0

# @numba.njit
def compute_W(
        alpha: double[:, ::1],
        beta: double[:, ::1],
        l1: int, l2: int, l3: int,
        r_values: double[::1],
        c_ell: double[::1]
    ):
    cdef cnp.double_t C6 = 6.0 * c_ell[l1] * c_ell[l2] * c_ell[l3]
    cdef cnp.double_t B = compute_bispectrum_local(alpha, beta, l1, l2, l3, r_values)
    return (2 * 3.14159) ** 2 * B / C6

class SIntegrand(vegas.LBatchIntegrand):
    def __init__(self, lmax, alpha, beta, rs, c_ell):
        self.lmax = lmax
        self.alpha = np.ascontiguousarray(alpha)
        self.beta = np.ascontiguousarray(beta)
        self.rs = np.ascontiguousarray(rs)
        self.c_ell = np.ascontiguousarray(c_ell)
        self.c_ell_size = self.c_ell.shape[0]

    def __call__(self, x):
        cdef:
            int batch_size = x.shape[0]
            double[::1] results = np.zeros(batch_size, dtype=np.float64)
            int i, l1_mag, l2_mag, l3_mag
            double[::1] l1_vec = np.zeros(2, dtype=np.float64)
            double[::1] l2_vec = np.zeros(2, dtype=np.float64)
            double[::1] l3_vec = np.zeros(2, dtype=np.float64)
            double W, B

        for i in range(batch_size):
            # Extract vectors for this point (can't use cdef inside loop)
            l1_vec[0] = x[i, 0]
            l1_vec[1] = x[i, 1]
            l2_vec[0] = x[i, 2]
            l2_vec[1] = x[i, 3]

            # Calculate l3 vector
            l3_vec[0] = -l1_vec[0] - l2_vec[0]
            l3_vec[1] = -l1_vec[1] - l2_vec[1]

            # Validity checks
            if (abs(l3_vec[0]) > self.lmax or abs(l3_vec[1]) > self.lmax or
                abs(l3_vec[0]) <= 2 or abs(l3_vec[1]) <= 2):
                continue  # Skip this point

            # Calculate magnitudes
            l1_mag = mag(l1_vec, l1_vec)
            l2_mag = mag(l2_vec, l2_vec)
            l3_mag = mag(l3_vec, l3_vec)

            if not (2 <= l1_mag <= self.c_ell_size and
                    2 <= l2_mag <= self.c_ell_size and
                    2 <= l3_mag <= self.c_ell_size):
                continue  # Skip this point

            # Compute result for this point
            W = compute_W(
                self.alpha, self.beta, l1_mag, l2_mag, l3_mag, self.rs, self.c_ell
            )
            B = compute_bispectrum_local(
                self.alpha, self.beta, l1_mag, l2_mag, l3_mag, self.rs
            )
            results[i] = W * B

        return results


def compute_S(alpha, beta, c_ell, radii, lmax, neval=(1e5, 1e5), nitn=(10, 10), summary=False, n_cpus=1):
    integrand = SIntegrand(lmax, np.ascontiguousarray(alpha), np.ascontiguousarray(beta), np.ascontiguousarray(radii), np.ascontiguousarray(c_ell))
    integ = vegas.Integrator([[-lmax, lmax], [-lmax, lmax]] * 2)

    integ(
        integrand,
        nitn=nitn[0],
        neval=neval[0],
        nproc=n_cpus,
    )
    result = integ(
        integrand,
        nitn=nitn[1],
        neval=neval[1],
        nproc=n_cpus,
    )

    if summary:
        print(result.summary(), flush=True)

    return result.mean / (2 * np.pi) ** 4 / np.pi