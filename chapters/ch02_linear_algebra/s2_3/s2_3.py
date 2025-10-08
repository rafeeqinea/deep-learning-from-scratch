"""
Section 2.3 — Identity and Inverse (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .identity import (
    eye,        # identity matrix I_n
    projector,  # U U^T when columns of U are orthonormal
)
from .inverse_solve import (
    is_invertible,  # rank-based check for invertibility
    solve,          # prefer solve(A, b) over inverse
    maybe_inverse,  # guarded inverse with condition-number check
)

__all__ = [
    "eye", "projector",
    "is_invertible", "solve", "maybe_inverse",
]
