# core/linear_algebra/span.py
from typing import List
from .matrices import Matrix, n_rows, n_cols, transpose
from .vectors import Vector

def rank(A: Matrix) -> int:
    """
    Computes the rank of a matrix using Gaussian elimination.
    The rank is the number of pivots found.
    """
    m, n = n_rows(A), n_cols(A)
    # Work on a copy
    A_copy = [row[:] for row in A]
    
    rank_val = 0
    pivot_row = 0
    for j in range(n): # Iterate through columns
        if pivot_row == m:
            break
            
        # Find a row with a non-zero entry in this column to be the pivot
        i = pivot_row
        while i < m and abs(A_copy[i][j]) < 1e-12:
            i += 1
            
        if i < m: # Pivot found
            # Swap rows to move pivot to pivot_row
            A_copy[pivot_row], A_copy[i] = A_copy[i], A_copy[pivot_row]
            pivot_val = A_copy[pivot_row][j]
            
            # Eliminate other entries in this column below the pivot
            for i in range(pivot_row + 1, m):
                factor = A_copy[i][j] / pivot_val
                for k in range(j, n):
                    A_copy[i][k] -= factor * A_copy[pivot_row][k]
            
            pivot_row += 1
            
    return pivot_row

def are_linearly_independent(vectors: List[Vector]) -> bool:
    """
    Checks if a set of vectors is linearly independent.
    Vectors are independent if the rank of the matrix formed by them
    is equal to the number of vectors.
    """
    if not vectors:
        return True # The empty set is trivially independent.

    # All vectors must have the same dimension
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("All vectors must have the same dimension.")

    # If there are more vectors than dimensions, they must be dependent.
    if len(vectors) > dim:
        return False
        
    # Create a matrix where the vectors are the columns, then find its rank.
    # To do this, we treat the list of vectors as rows and transpose it.
    matrix = transpose(vectors)
    
    return rank(matrix) == len(vectors)