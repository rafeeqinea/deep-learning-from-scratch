# core/linear_algebra/linear_systems.py
from typing import List, Union
from .matrices import Matrix, n_rows, n_cols
from .vectors import Vector

Number = Union[int, float]

def solve(A: Matrix, b: Vector) -> Vector:
    """
    Solves the linear system Ax = b for x using Gaussian elimination
    with partial pivoting.
    """
    # --- 1. Validation ---
    m, n = n_rows(A), n_cols(A)
    if m != n:
        raise ValueError("Matrix A must be square to solve the system.")
    if m != len(b):
        raise ValueError(f"Shape mismatch: A has {m} rows but b has {len(b)} elements.")

    # --- 2. Create an augmented matrix [A|b] ---
    # We work on a copy to avoid modifying the original matrix A.
    aug = [row_A + [b_i] for row_A, b_i in zip(A, b)]

    # --- 3. Forward Elimination with Partial Pivoting ---
    for i in range(n):
        # Find the pivot row (the one with the largest element in the current column)
        pivot_row = i
        for j in range(i + 1, n):
            if abs(aug[j][i]) > abs(aug[pivot_row][i]):
                pivot_row = j
        
        # Swap the current row with the pivot row
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        # Check for singularity. If the pivot element is close to zero,
        # the matrix is singular or ill-conditioned.
        pivot_val = aug[i][i]
        if abs(pivot_val) < 1e-12:
            raise ValueError("Matrix is singular; the system may not have a unique solution.")

        # Eliminate the column in rows below the pivot
        for j in range(i + 1, n):
            factor = aug[j][i] / pivot_val
            for k in range(i, n + 1):
                aug[j][k] -= factor * aug[i][k]

    # --- 4. Back Substitution ---
    x: List[Number] = [0] * n
    for i in range(n - 1, -1, -1):
        # Start with the constant term
        val = aug[i][n]
        # Subtract the terms for variables we've already solved
        for j in range(i + 1, n):
            val -= aug[i][j] * x[j]
        # Divide by the diagonal element to solve for x[i]
        x[i] = val / aug[i][i]
        
    return x