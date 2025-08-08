"""
Code largely taken from https://github.com/albakalaja/CMB_fisher_bispectrum/blob/main/scripts/compute_fisher.pyx
Thank you: Alba Kalaja

Converted to a class-based implementation for better reuse.
Extended with trispectrum calculation functionality.
"""

import numpy as np
from scipy.integrate import trapezoid
cimport cython

def compute_bispectrum(alpha, beta, l1, l2, l3, r_values):
    r_np = np.asarray(r_values, dtype=np.double)
    alpha_np = np.asarray(alpha, dtype=np.double)
    beta_np = np.asarray(beta, dtype=np.double)

    integrand = r_np ** 2 * (
        beta_np[l3] * beta_np[l2] * alpha_np[l1] +
        beta_np[l1] * beta_np[l3] * alpha_np[l2] +
        beta_np[l2] * beta_np[l1] * alpha_np[l3]
    )
    
    # Compute the bispectrum using trapezoid integration
    return 6.0/5.0 * trapezoid(integrand, x=r_np)

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