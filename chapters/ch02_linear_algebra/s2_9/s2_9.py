"""
Section 2.9 — Moore–Penrose Pseudoinverse (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .pseudoinverse import (
    pinv_svd,       # pseudoinverse via SVD with rcond threshold
    min_norm_solve, # x = A^+ b  (minimum-norm least-squares solution)
)

__all__ = [
    "pinv_svd", "min_norm_solve",
]
