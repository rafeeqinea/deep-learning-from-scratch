# core/linear_algebra/identity_zeros_ones.py
from typing import Union
from .matrices import Matrix, create_matrix

Number = Union[int, float]

def identity(n: int) -> Matrix:
    """
    Creates an n x n identity matrix.
    The identity matrix has 1s on the main diagonal and 0s everywhere else.
    """
    if n <= 0:
        raise ValueError("Size 'n' for an identity matrix must be a positive integer.")
    
    # For each element (i, j), place a 1 if i == j, otherwise place a 0.
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def zeros(rows: int, cols: int) -> Matrix:
    """
    Creates a matrix of a given size filled entirely with zeros.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Number of rows and columns must be positive integers.")
    
    return [[0 for _ in range(cols)] for _ in range(rows)]

def ones(rows: int, cols: int) -> Matrix:
    """
    Creates a matrix of a given size filled entirely with ones.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Number of rows and columns must be positive integers.")
        
    return [[1 for _ in range(cols)] for _ in range(rows)]