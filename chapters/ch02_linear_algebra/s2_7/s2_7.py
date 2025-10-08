"""
Section 2.7 — Eigendecomposition (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .eigendecomp import (
    eig_sym,             # eigenpairs of a (nearly) symmetric matrix
    spectral_decomp,     # eigendecomposition with gentle symmetrization
    is_positive_definite # check PD (all eigenvalues > tol)
)

__all__ = [
    "eig_sym", "spectral_decomp", "is_positive_definite",
]
