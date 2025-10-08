"""
Section 2.8 — Singular Value Decomposition (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .svd_tools import (
    svd_thin,       # thin SVD (U, S, VT) with full_matrices=False
    cond_number,    # spectral condition number: sigma_max / sigma_min
    low_rank_approx # rank-r approximation via top singular triplets
)

__all__ = [
    "svd_thin", "cond_number", "low_rank_approx",
]
