"""
Section 2.2 — Multiplying Matrices and Vectors (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .matvec import (
    mm,          # matrix × matrix
    mv,          # matrix × vector
    vv,          # vector · vector (dot)
    safe_matmul, # guarded matmul with helpful errors
)
from .linear_comb import (
    lincomb,     # linear combination of vectors
    affine_map,  # X @ W + b
    add_affine,  # Y + b (row-wise bias add)
)

__all__ = [
    "mm", "mv", "vv", "safe_matmul",
    "lincomb", "affine_map", "add_affine",
]
