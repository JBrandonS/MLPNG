"""
MLPNG Bispectrum module for computing CMB local bispectra and trispectra.
"""

# Import these after build
try:
    from .bispectrum import (
        compute_bispectrum,
        compute_trispectrum,
    )

    __all__ = [
        "compute_bispectrum",
        "compute_trispectrum",
    ]
except ImportError:
    # Module might not be built yet
    __all__ = []
