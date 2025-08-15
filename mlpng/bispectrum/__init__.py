"""
MLPNG Bispectrum module for computing CMB local bispectra and trispectra.
"""

# Import these after build
try:
    from .bispectrum import (
        compute_bispectrum,
        compute_trispectrum,
        valid_triangle,
        tri,
        SIntegrand,
        N2Integrand,
        DeltaN2Integrand,
        SN2Integrand,
        SN2NTIntegrand,
    )

    __all__ = [
        "compute_bispectrum",
        "compute_trispectrum",
        "valid_triangle",
        "tri",
        "SIntegrand",
        "N2Integrand",
        "DeltaN2Integrand",
        "SN2Integrand",
        "SN2NTIntegrand",
    ]
except ImportError:
    # Module might not be built yet
    __all__ = []
