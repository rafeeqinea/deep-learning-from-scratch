"""
Section 2.11 — Determinant (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .determinant import (
    det,              # determinant of a square matrix
    logdet_safe,      # numerically stable log(det(A)) for PD / sign>0
    sign_logabsdet,   # returns (sign, log|det(A)|) from slogdet
    volume_scale,     # |det(A)| as a positive "volume scale" factor
)

__all__ = [
    "det", "logdet_safe", "sign_logabsdet", "volume_scale",
]
