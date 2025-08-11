# core/linear_algebra/matrices.py
from typing import Iterable, List, Tuple, Union

Number = Union[int, float]
Matrix = List[List[Number]]

def create_matrix(rows: Iterable[Iterable[Number]]) -> Matrix:
    """
    Build a rectangular numeric matrix (list of lists).
    """
    A: Matrix = [list(r) for r in rows]
    if not A:
        raise ValueError("Matrix must have at least one row.")
    ncols = len(A[0])
    if ncols == 0:
        raise ValueError("Matrix must have at least one column.")
    for i, row in enumerate(A):
        if len(row) != ncols:
            raise ValueError(f"Non-rectangular matrix: row 0 has length {ncols} but row {i} has length {len(row)}.")
        for j, x in enumerate(row):
            if not isinstance(x, (int, float)):
                raise TypeError(f"Matrix entries must be int or float, got {type(x).__name__} at ({i},{j}).")
    return A

def shape(A: Matrix) -> Tuple[int, int]:
    """Return (m, n) where m=#rows and n=#cols."""
    _validate_matrix(A)
    return len(A), len(A[0])

def n_rows(A: Matrix) -> int:
    """Number of rows."""
    _validate_matrix(A)
    return len(A)

def n_cols(A: Matrix) -> int:
    """Number of columns."""
    _validate_matrix(A)
    return len(A[0])

def transpose(A: Matrix) -> Matrix:
    """Return the transpose of A."""
    _validate_matrix(A)
    return [list(col) for col in zip(*A)]

def add_matrices(A: Matrix, B: Matrix) -> Matrix:
    """Elementwise A + B."""
    _validate_same_shape(A, B, "addition")
    return [[a + b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)]

def subtract_matrices(A: Matrix, B: Matrix) -> Matrix:
    """Elementwise A - B."""
    _validate_same_shape(A, B, "subtraction")
    return [[a - b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)]

def scale_matrix(scalar: Number, A: Matrix) -> Matrix:
    """Scalar multiply: scalar * A."""
    if not isinstance(scalar, (int, float)):
        raise TypeError(f"Scalar must be int or float, got {type(scalar).__name__}.")
    _validate_matrix(A)
    return [[scalar * x for x in row] for row in A]

# --- internal helpers ---
def _validate_matrix(A: Matrix) -> None:
    if not isinstance(A, list) or not A or not isinstance(A[0], list):
        raise ValueError("Matrix must be a non-empty list of non-empty lists.")
    ncols = len(A[0])
    if ncols == 0:
        raise ValueError("Matrix must have at least one column.")
    for i, row in enumerate(A):
        if len(row) != ncols:
            raise ValueError(f"Non-rectangular matrix: row 0 has length {ncols} but row {i} has length {len(row)}.")
        for j, x in enumerate(row):
            if not isinstance(x, (int, float)):
                raise TypeError(f"Matrix entries must be int or float, got {type(x).__name__} at ({i},{j}).")

def _validate_same_shape(A: Matrix, B: Matrix, op_name: str) -> None:
    _validate_matrix(A)
    _validate_matrix(B)
    ra, ca = shape(A)
    rb, cb = shape(B)
    if (ra, ca) != (rb, cb):
        raise ValueError(f"Shape mismatch for {op_name}: A is {ra}x{ca} but B is {rb}x{cb}.")
