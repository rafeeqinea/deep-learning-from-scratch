"""
Section 2.10 — Trace (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .trace_ops import (
    trace,               # trace of a square matrix
    trace_cyclic_equal,  # check tr(AB) = tr(BA) or tr(ABC) = tr(BCA) = tr(CAB)
)

__all__ = [
    "trace", "trace_cyclic_equal",
]
