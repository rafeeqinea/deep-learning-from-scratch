"""
Section 2.5 — Norms (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .norms import (
    norm_l1,       # vector L1 (sum of absolute values)
    norm_l2,       # vector L2 (Euclidean)
    norm_linf,     # vector L-infinity (max absolute entry)
    frobenius,     # matrix Frobenius norm
    op_norm_2,     # matrix operator 2-norm (largest singular value)
)

__all__ = [
    "norm_l1", "norm_l2", "norm_linf",
    "frobenius", "op_norm_2",
]
