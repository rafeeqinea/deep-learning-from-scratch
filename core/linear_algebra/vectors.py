# core/linear_algebra/vectors.py
from typing import Iterable, List, Union
import math

Number = Union[int, float]
Vector = List[Number]

def create_vector(elements: Iterable[Number]) -> Vector:
    """
    Create a vector from iterable of numbers.
    Validates non-empty and numeric.
    """
    v = list(elements)
    if not v:
        raise ValueError("Vector must have at least one element.")
    for i, x in enumerate(v):
        if not isinstance(x, (int, float)):
            raise TypeError(f"Vector elements must be int or float, got {type(x).__name__} at index {i}.")
    return v

def add_vectors(a: Vector, b: Vector) -> Vector:
    """Elementwise vector addition."""
    _validate_same_length(a, b, "addition")
    return [x + y for x, y in zip(a, b)]

def subtract_vectors(a: Vector, b: Vector) -> Vector:
    """Elementwise vector subtraction."""
    _validate_same_length(a, b, "subtraction")
    return [x - y for x, y in zip(a, b)]

def scalar_multiply(scalar: Number, v: Vector) -> Vector:
    """Multiply vector by scalar."""
    if not isinstance(scalar, (int, float)):
        raise TypeError(f"Scalar must be int or float, got {type(scalar).__name__}.")
    _validate_vector(v)
    return [scalar * x for x in v]

def dot_product(a: Vector, b: Vector) -> Number:
    """Dot product of two vectors."""
    _validate_same_length(a, b, "dot product")
    return sum(x * y for x, y in zip(a, b))

def vector_norm(v: Vector, p: Union[int, str] = 2) -> float:
    """
    Compute p-norm of vector.
    p can be 1, 2, or "inf".
    """
    _validate_vector(v)
    if p == 1:
        return float(sum(abs(x) for x in v))
    elif p == 2:
        return math.sqrt(sum(x * x for x in v))
    elif p == "inf":
        return float(max(abs(x) for x in v))
    else:
        raise ValueError(f"Unsupported norm type: {p}")

def normalize_vector(v: Vector) -> Vector:
    """Return unit vector in direction of v."""
    norm = vector_norm(v, 2)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")
    return [x / norm for x in v]

# --- internal validation ---
def _validate_vector(v: Vector) -> None:
    if not isinstance(v, list) or not v:
        raise ValueError("Vector must be a non-empty list.")
    for i, x in enumerate(v):
        if not isinstance(x, (int, float)):
            raise TypeError(f"Vector elements must be int or float, got {type(x).__name__} at index {i}.")

def _validate_same_length(a: Vector, b: Vector, op_name: str) -> None:
    _validate_vector(a)
    _validate_vector(b)
    if len(a) != len(b):
        raise ValueError(f"Vectors must be same length for {op_name}. Got {len(a)} and {len(b)}.")
