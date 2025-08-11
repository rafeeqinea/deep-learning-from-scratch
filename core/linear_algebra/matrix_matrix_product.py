# core/linear_algebra/matrix_matrix_product.py
from typing import Union

# Import all the tools you've already built
from core.linear_algebra.matrices import Matrix, _validate_matrix, n_cols, n_rows, transpose
from core.linear_algebra.vectors import dot_product

Number = Union[int, float]

def matrix_product(A: Matrix, B: Matrix) -> Matrix:
    """
    Computes the product of two matrices, C = A * B.
    If A is (m, n) and B is (n, p), the result C is (m, p).
    """
    # 1. Validate inputs
    _validate_matrix(A)
    _validate_matrix(B)

    # 2. Check for shape compatibility (inner dimensions must match)
    if n_cols(A) != n_rows(B):
        raise ValueError(
            f"Shape mismatch for matrix product: "
            f"A has {n_cols(A)} columns but B has {n_rows(B)} rows."
        )

    # 3. Use transpose on B to easily access its columns
    B_T = transpose(B)

    # 4. Compute the product using a nested list comprehension.
    # For each row in A, compute its dot product with each column of B (which is now a row in B_T).
    return [
        [dot_product(row_A, col_B) for col_B in B_T]
        for row_A in A
    ]