"""
Section 2.1 — Scalars, Vectors, Matrices and Tensors (NumPy-only).

This file simply re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .shapes import (
    infer_shape,
    ensure_vector,
    ensure_matrix,
    ensure_tensor_nd,
)
from .dtypes import (
    as_float32,
    as_float64,
    is_integer_array,
)
from .basic_validators import (
    check_same_shape,
    check_broadcastable,
    check_finite,
)

# This tells "from s2_1 import *" which names are public
__all__ = [
    # shapes
    "infer_shape", "ensure_vector", "ensure_matrix", "ensure_tensor_nd",
    # dtypes
    "as_float32", "as_float64", "is_integer_array",
    # validators
    "check_same_shape", "check_broadcastable", "check_finite",
]
