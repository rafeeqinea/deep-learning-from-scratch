"""
Section 2.6 — Special Matrices and Vectors (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .special import (
    is_diagonal,      # off-diagonal entries are (near) zero
    is_symmetric,     # A == A^T (within tolerance)
    is_orthogonal,    # Q^T Q == I (within tolerance)
    is_psd,           # symmetric positive semidefinite
)

__all__ = [
    "is_diagonal", "is_symmetric", "is_orthogonal", "is_psd",
]
