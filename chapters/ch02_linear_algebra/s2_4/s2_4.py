"""
Section 2.4 — Linear Dependence and Span (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .span_rank import (
    rank,           # numerical rank via SVD
    is_full_rank,   # True if rank(A) == min(m, n)
    span_contains,  # check if a vector v lies in span of given basis
)
from .basis import (
    orthonormal_basis,  # orthonormal basis for the column space of A
)

__all__ = [
    "rank", "is_full_rank", "span_contains",
    "orthonormal_basis",
]
