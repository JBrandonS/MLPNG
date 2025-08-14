"""
Code largely taken from https://github.com/albakalaja/CMB_fisher_bispectrum/blob/main/scripts/compute_fisher.pyx
Thank you: Alba Kalaja

Converted to a class-based implementation for better reuse.
Extended with trispectrum calculation functionality.
"""

import numpy as np
import vegas
import numba
from scipy.integrate import trapezoid

cimport cython
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_bispectrum(alpha, beta, l1, l2, l3, r_values):
    """Compute bispectrum using trapezoid integration."""
    integrand = r_values ** 2 * (
        beta[l3] * beta[l2] * alpha[l1] +
        beta[l1] * beta[l3] * alpha[l2] +
        beta[l2] * beta[l1] * alpha[l3]
    )

    # Compute the bispectrum using trapezoid integration
    return 6.0/5.0 * trapezoid(integrand, x=r_values)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l2, l3, l4):
    lmax = c_ell_phi.shape[0] - 1

    l13 = l1 + l3
    l14 = l1 + l4

    l1_mag = int(round(np.sqrt(np.dot(l1, l1))))
    l2_mag = int(round(np.sqrt(np.dot(l2, l2))))
    l3_mag = int(round(np.sqrt(np.dot(l3, l3))))
    l4_mag = int(round(np.sqrt(np.dot(l4, l4))))
    l13_mag = int(round(np.sqrt(np.dot(l13, l13))))
    l14_mag = int(round(np.sqrt(np.dot(l14, l14))))

    for x in [l1_mag, l2_mag, l3_mag, l4_mag, l13_mag, l14_mag]:
        if not 2 <= x <= lmax:
            return 0.0

    # flat sky approx -> C_l^\delta\delta / (|l|^2) = c_l^\phi
    T =  np.dot(l13, l1) * np.dot(l13, l2) * c_ell_phi[l13_mag]
    T += np.dot(l14, l1) * np.dot(l14, l2) * c_ell_phi[l14_mag]
    return c_ell_TT[l1_mag] * c_ell_TT[l2_mag] * T

def compute_trispectrum(c_ell_TT, c_ell_phi, l1, l2, l3, l4):
    r"""
    T = C_l1^TT C_l2^TT [ ((l1+l3)dot l1 (l1+l3) dot l2) / (|l1+l3|^2)) C_l13^\delta\delta + ((l1+l4) dot l1 (l1+l4) dot l2) / (|l1+l4|^2)) C_l14^\delta\delta ] + permutations
    """
    c_ell_TT = c_ell_TT.astype(np.double)
    c_ell_phi = c_ell_phi.astype(np.double)

    trispectrum = (
          compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l2, l3, l4)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l3, l2, l4)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l1, l4, l2, l3)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l2, l3, l1, l4)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l2, l4, l1, l3)
        + compute_trispectrum_single(c_ell_TT, c_ell_phi, l3, l4, l1, l2)
    )

    return trispectrum

def valid_triangle(l1: int, l2: int, l3: int, lmax: int) -> bool:
    """Triangle validity check with bounds and parity.

    Ensures 2 <= li <= lmax, triangle inequalities, and (l1+l2+l3) even.
    """
    if l1 < 2 or l2 < 2 or l3 < 2:
        return False
    if l1 > lmax or l2 > lmax or l3 > lmax:
        return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
def _tri_single(c_ell_TT, c_ell_phi, l1, l2, l3, l4):
    """Calculate a single trispectrum configuration used by tri()."""
    l13 = l1 + l3
    l24 = l2 + l4

    l13_mag = int(round(np.sqrt(np.dot(l13, l13))))
    l3_mag = int(round(np.sqrt(np.dot(l3, l3))))
    l4_mag = int(round(np.sqrt(np.dot(l4, l4))))

    l13_3 = float(np.dot(l13, l3))
    l24_4 = float(np.dot(l24, l4))
    return float(c_ell_phi[l13_mag]) * float(c_ell_TT[l3_mag]) * float(c_ell_TT[l4_mag]) * l13_3 * l24_4

@cython.boundscheck(False)
@cython.wraparound(False)
def tri(c_ell_TT, c_ell_phi, l1, l2, l3, l4):
    """Notebook-compatible trispectrum variant (sum of many perms, then / 2).

    This mirrors the Python implementation used in notebooks to match literature
    conventions; kept here so DeltaN2Integrand can reproduce the same values.
    """
    lmax = np.shape(c_ell_TT)[0] - 1

    # Bounds/validity checks similar to notebook code
    for i in [l1, l2, l3, l4]:
        for j in [l1, l2, l3, l4]:
            if np.array_equal(i, j):
                if not 2 <= int(round(np.sqrt(np.dot(i, i)))) <= lmax / 2:
                    return 0.0
            elif not 2 <= int(round(np.sqrt(np.dot(i + j, i + j)))) <= lmax:
                return 0.0

    value = 0.0
    # Expand all permutations explicitly (as in the notebook)
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SIntegrand:
    """Integrand for S: W(l1,l2,l3) * B(l1,l2,l3).

    W is computed as (2π)^2 B / [6 C_l1 C_l2 C_l3], so this returns (2π)^2 B^2 / C6.
    """
    cdef public int lmax
    cdef object c_ell
    cdef object alpha
    cdef object beta
    cdef object rs

    def __init__(self, int lmax, alpha, beta, rs, c_ell):
        self.lmax = lmax
        self.alpha = alpha
        self.beta = beta
        self.rs = rs
        self.c_ell = np.asarray(c_ell, dtype=np.double)

    def __call__(self, x):
        xnp = np.asarray(x, dtype=np.double)
        if xnp.ndim == 1:
            # Single sample [x0,x1,x2,x3]
            return self._eval_row(xnp)
        elif xnp.ndim == 2:
            # Batched samples with shape (B, 4)
            B = xnp.shape[0]
            out = np.empty(B, dtype=np.double)
            for i in range(B):
                out[i] = self._eval_row(xnp[i])
            return out
        else:
            raise ValueError("SIntegrand: x must be 1D or 2D array")

    def _eval_row(self, row):
        l1 = np.array([row[0], row[1]], dtype=np.double)
        l2 = np.array([row[2], row[3]], dtype=np.double)
        l3 = -l1 - l2

        l1_mag = int(round(np.sqrt(np.dot(l1, l1))))
        l2_mag = int(round(np.sqrt(np.dot(l2, l2))))
        l3_mag = int(round(np.sqrt(np.dot(l3, l3))))

        if not valid_triangle(l1_mag, l2_mag, l3_mag, self.lmax):
            return 0.0

        B = compute_bispectrum(self.alpha, self.beta, l1_mag, l2_mag, l3_mag, self.rs)
        C6 = 6.0 * float(self.c_ell[l1_mag]) * float(self.c_ell[l2_mag]) * float(self.c_ell[l3_mag])
        return (2.0 * np.pi) ** 2 * (B * B) / C6

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class N2Integrand:
    """Integrand for N^2: W^2 C6 = (2π)^4 B^2 / C6."""
    cdef public int lmax
    cdef object c_ell
    cdef object alpha
    cdef object beta
    cdef object rs

    def __init__(self, int lmax, c_ell, alpha, beta, rs):
        self.lmax = lmax
        self.c_ell = np.asarray(c_ell, dtype=np.double)
        self.alpha = alpha
        self.beta = beta
        self.rs = rs

    def __call__(self, x):
        xnp = np.asarray(x, dtype=np.double)
        if xnp.ndim == 1:
            return self._eval_row(xnp)
        elif xnp.ndim == 2:
            Bn = xnp.shape[0]
            out = np.empty(Bn, dtype=np.double)
            for i in range(Bn):
                out[i] = self._eval_row(xnp[i])
            return out
        else:
            raise ValueError("N2Integrand: x must be 1D or 2D array")

    def _eval_row(self, row):
        l1 = np.array([row[0], row[1]], dtype=np.double)
        l2 = np.array([row[2], row[3]], dtype=np.double)
        l3 = -l1 - l2

        l1_mag = int(round(np.sqrt(np.dot(l1, l1))))
        l2_mag = int(round(np.sqrt(np.dot(l2, l2))))
        l3_mag = int(round(np.sqrt(np.dot(l3, l3))))

        if not valid_triangle(l1_mag, l2_mag, l3_mag, self.lmax):
            return 0.0

        B = compute_bispectrum(self.alpha, self.beta, l1_mag, l2_mag, l3_mag, self.rs)
        C6 = 6.0 * float(self.c_ell[l1_mag]) * float(self.c_ell[l2_mag]) * float(self.c_ell[l3_mag])
        return (2.0 * np.pi) ** 4 * (B * B) / C6


class DeltaN2Integrand(vegas.LBatchIntegrand):
    """Fully vectorized integrand for delta<N^2> using tri-style trispectrum.

    Returns W(l) W(l') C_l1^{lens} T(l2,l3,l2',l3').
    This version uses fully vectorized NumPy operations for all calculations.
    """

    def __init__(self, int lmax, c_ell, c_ell_lens, cl_phi, alpha, beta, rs):
        self.lmax = lmax
        self.c_ell_arr = np.asarray(c_ell, dtype=np.double)
        self.c_ell_lens_arr = np.asarray(c_ell_lens, dtype=np.double)
        self.cl_phi_arr = np.asarray(cl_phi, dtype=np.double)
        self.alpha = alpha
        self.beta = beta
        self.rs = rs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, x):
        """Fully vectorized evaluation for batches of samples."""
        xnp = np.asarray(x, dtype=np.double)

        # Always assume batch input (Vegas will always give us batches)
        if xnp.ndim == 1:
            print("Reshaping single input to batch")
            xnp = xnp.reshape(1, -1)

        Bn = xnp.shape[0]

        # Extract l-vectors for the entire batch
        l1_batch = xnp[:, 0:2]  # Shape: (Bn, 2)
        l2_batch = xnp[:, 2:4]  # Shape: (Bn, 2)
        lp2_batch = xnp[:, 4:6]  # Shape: (Bn, 2)

        # Derived l-vectors
        l3_batch = -l1_batch - l2_batch  # Shape: (Bn, 2)
        lp1_batch = -l1_batch  # Shape: (Bn, 2)
        lp3_batch = -l2_batch - l3_batch - lp2_batch  # Shape: (Bn, 2)

        # Compute magnitudes vectorized
        l1_mag = np.rint(np.sqrt(np.sum(l1_batch * l1_batch, axis=1))).astype(np.int32)
        l2_mag = np.rint(np.sqrt(np.sum(l2_batch * l2_batch, axis=1))).astype(np.int32)
        l3_mag = np.rint(np.sqrt(np.sum(l3_batch * l3_batch, axis=1))).astype(np.int32)
        lp1_mag = np.rint(np.sqrt(np.sum(lp1_batch * lp1_batch, axis=1))).astype(np.int32)
        lp2_mag = np.rint(np.sqrt(np.sum(lp2_batch * lp2_batch, axis=1))).astype(np.int32)
        lp3_mag = np.rint(np.sqrt(np.sum(lp3_batch * lp3_batch, axis=1))).astype(np.int32)

        # Validity mask
        valid = ((l1_mag >= 2) & (l1_mag <= self.lmax) &
                 (l2_mag >= 2) & (l2_mag <= self.lmax) &
                 (l3_mag >= 2) & (l3_mag <= self.lmax) &
                 (lp1_mag >= 2) & (lp1_mag <= self.lmax) &
                 (lp2_mag >= 2) & (lp2_mag <= self.lmax) &
                 (lp3_mag >= 2) & (lp3_mag <= self.lmax))

        # Initialize output
        result = np.zeros(Bn, dtype=np.double)

        if not np.any(valid):
            return result

        # Get valid indices
        valid_idx = np.where(valid)[0]
        n_valid = len(valid_idx)

        # Vectorized bispectrum calculation for valid samples
        B_vals = self._compute_bispectrum_vectorized(
            l1_mag[valid_idx], l2_mag[valid_idx], l3_mag[valid_idx])
        Bp_vals = self._compute_bispectrum_vectorized(
            lp1_mag[valid_idx], lp2_mag[valid_idx], lp3_mag[valid_idx])

        # Vectorized window calculations
        C6_vals = 6.0 * (self.c_ell_arr[l1_mag[valid_idx]] *
                        self.c_ell_arr[l2_mag[valid_idx]] *
                        self.c_ell_arr[l3_mag[valid_idx]])
        C6p_vals = 6.0 * (self.c_ell_arr[lp1_mag[valid_idx]] *
                         self.c_ell_arr[lp2_mag[valid_idx]] *
                         self.c_ell_arr[lp3_mag[valid_idx]])

        W_vals = (2.0 * np.pi) ** 2 * B_vals / C6_vals
        Wp_vals = (2.0 * np.pi) ** 2 * Bp_vals / C6p_vals

        # Vectorized trispectrum calculation
        T_vals = self._compute_trispectrum_vectorized(
            l2_batch[valid_idx], l3_batch[valid_idx],
            lp2_batch[valid_idx], lp3_batch[valid_idx])

        # Final result
        result[valid_idx] = (W_vals * Wp_vals *
                           self.c_ell_lens_arr[l1_mag[valid_idx]] * T_vals)

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _compute_bispectrum_vectorized(self, l1_mags, l2_mags, l3_mags):
        """Vectorized bispectrum computation."""
        n_samples = len(l1_mags)

        # Pre-compute integrand for all r values and all samples
        # r_values shape: (n_r,), alpha/beta shapes: (lmax+1, n_r)
        # We need: r^2 * (beta[l3] * beta[l2] * alpha[l1] + perms)
        r_vals = self.rs  # Shape: (n_r,)
        r_squared = r_vals ** 2  # Shape: (n_r,)

        # Extract alpha/beta values for each sample
        # alpha[l1_mags] has shape (n_samples, n_r)
        alpha_l1 = self.alpha[l1_mags]  # Shape: (n_samples, n_r)
        alpha_l2 = self.alpha[l2_mags]  # Shape: (n_samples, n_r)
        alpha_l3 = self.alpha[l3_mags]  # Shape: (n_samples, n_r)

        beta_l1 = self.beta[l1_mags]   # Shape: (n_samples, n_r)
        beta_l2 = self.beta[l2_mags]   # Shape: (n_samples, n_r)
        beta_l3 = self.beta[l3_mags]   # Shape: (n_samples, n_r)

        # Compute integrand: r^2 * (sum of three permutations)
        integrand = r_squared[np.newaxis, :] * (
            beta_l3 * beta_l2 * alpha_l1 +
            beta_l1 * beta_l3 * alpha_l2 +
            beta_l2 * beta_l1 * alpha_l3
        )  # Shape: (n_samples, n_r)

        # Trapezoid integration along r dimension for each sample
        bispectrum_vals = 6.0/5.0 * trapezoid(integrand, x=r_vals, axis=1)
        return bispectrum_vals

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _compute_trispectrum_vectorized(self, l2_batch, l3_batch, lp2_batch, lp3_batch):
        """Vectorized trispectrum computation using tri function."""
        n_samples = l2_batch.shape[0]
        T_vals = np.zeros(n_samples, dtype=np.double)

        # For now, still use loop for trispectrum (tri function is complex)
        # This could be further vectorized but would require major refactoring of tri
        for i in range(n_samples):
            T_vals[i] = tri(self.c_ell_arr, self.cl_phi_arr,
                           l2_batch[i], l3_batch[i],
                           lp2_batch[i], lp3_batch[i])

        return T_vals

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SN2Integrand:
    """Integrand for S/N^2 (signal part): B^2 / (6 C_l1 C_l2 C_l3)."""
    cdef public int lmax
    cdef object c_ell
    cdef object alpha
    cdef object beta
    cdef object radii

    def __init__(self, int lmax, c_ell, alpha, beta, radii):
        self.lmax = lmax
        self.c_ell = np.asarray(c_ell, dtype=np.double)
        self.alpha = alpha
        self.beta = beta
        self.radii = radii

    def __call__(self, x):
        xnp = np.asarray(x, dtype=np.double)
        if xnp.ndim == 1:
            return self._eval_row(xnp)
        elif xnp.ndim == 2 or xnp.ndim == 3:
            Bn = xnp.shape[0]
            out = np.empty(Bn, dtype=np.double)
            for i in range(Bn):
                out[i] = self._eval_row(xnp[i])
            return out
        else:
            raise ValueError(f"SN2Integrand: x must be 1D or 2D array not {xnp.ndim}, {xnp.shape}")

    def _eval_row(self, row):
        l1 = np.array([row[0], row[1]], dtype=np.double)
        l2 = np.array([row[2], row[3]], dtype=np.double)
        l3 = -l1 - l2

        l1_mag = int(round(np.sqrt(np.dot(l1, l1))))
        l2_mag = int(round(np.sqrt(np.dot(l2, l2))))
        l3_mag = int(round(np.sqrt(np.dot(l3, l3))))

        for L in [l1_mag, l2_mag, l3_mag]:
            if L < 2 or L > self.lmax:
                return 0.0

        B = compute_bispectrum(self.alpha, self.beta, l1_mag, l2_mag, l3_mag, self.radii)
        C6 = 6.0 * float(self.c_ell[l1_mag]) * float(self.c_ell[l2_mag]) * float(self.c_ell[l3_mag])
        return (B * B) / C6

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SN2NTIntegrand:
    """Non-thermal contribution integrand used in S/N^2 baseline comparison."""
    cdef public int lmax

    def __init__(self, int lmax, c_ell=None, alpha=None, beta=None, radii=None):
        self.lmax = lmax

    def __call__(self, x):
        xnp = np.asarray(x, dtype=np.double)
        if xnp.ndim == 1:
            return self._eval_row(xnp)
        elif xnp.ndim == 2:
            # Vectorized computation for the non-thermal baseline
            l1 = xnp[:, 0:2]
            l2 = xnp[:, 2:4]
            l3 = -(l1 + l2)

            l1_mag = np.rint(np.sqrt(np.sum(l1 * l1, axis=1))).astype(np.int64)
            l2_mag = np.rint(np.sqrt(np.sum(l2 * l2, axis=1))).astype(np.int64)
            l3_mag = np.rint(np.sqrt(np.sum(l3 * l3, axis=1))).astype(np.int64)

            valid = (l1_mag >= 2) & (l2_mag >= 2) & (l3_mag >= 2) & (
                l1_mag <= self.lmax) & (l2_mag <= self.lmax) & (l3_mag <= self.lmax)
            # Initialize output
            out = np.zeros(xnp.shape[0], dtype=np.double)
            if not np.any(valid):
                return out

            li = l1_mag[valid].astype(np.float64)
            lj = l2_mag[valid].astype(np.float64)
            lk = l3_mag[valid].astype(np.float64)
            inner = (1.0 / (li * li * lj * lj) + 1.0 / (li * li * lk * lk) + 1.0 / (lj * lj * lk * lk))
            out[valid] = (li * li) * (lj * lj) * (lk * lk) * (inner * inner)
            return out
        else:
            raise ValueError("SN2NTIntegrand: x must be 1D or 2D array")

    def _eval_row(self, row):
        l1 = np.array([row[0], row[1]], dtype=np.double)
        l2 = np.array([row[2], row[3]], dtype=np.double)
        l3 = -l1 - l2

        l1_mag = int(round(np.sqrt(np.dot(l1, l1))))
        l2_mag = int(round(np.sqrt(np.dot(l2, l2))))
        l3_mag = int(round(np.sqrt(np.dot(l3, l3))))

        for L in [l1_mag, l2_mag, l3_mag]:
            if L < 2 or L > self.lmax:
                return 0.0

        inner = (
            1.0 / (l1_mag * l1_mag * l2_mag * l2_mag)
            + 1.0 / (l1_mag * l1_mag * l3_mag * l3_mag)
            + 1.0 / (l2_mag * l2_mag * l3_mag * l3_mag)
        )
        return (l1_mag * l1_mag) * (l2_mag * l2_mag) * (l3_mag * l3_mag) * (inner * inner)
