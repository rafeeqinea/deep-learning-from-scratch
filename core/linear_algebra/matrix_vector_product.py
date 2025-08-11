# core/linear_algebra/matrix_vector_product.py
from typing import Union

# Import necessary functions and types from your other modules
from core.linear_algebra.matrices import Matrix, n_cols, _validate_matrix
from core.linear_algebra.vectors import Vector, dot_product, _validate_vector

Number = Union[int, float]

def matrix_vector_product(A: Matrix, v: Vector) -> Vector:
    """
    Compute the matrix-vector product y = A * v.
    A must be m x n, v must have length n.
    Returns a vector of length m.
    """
    # 1. Validate the inputs first
    _validate_matrix(A)
    _validate_vector(v)
    
    # 2. Check for shape compatibility
    if n_cols(A) != len(v):
        raise ValueError(
            f"Shape mismatch for matrix-vector product: "
            f"A has {n_cols(A)} columns but vector has length {len(v)}."
        )

    # 3. Compute the product using a clean list comprehension
    #    by taking the dot product of each row of A with v.
    return [dot_product(row, v) for row in A]