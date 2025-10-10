# Rank and span utilities (NumPy only).
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix, ensure_vector,
)


def rank(A):
    """
    Numerical rank of A (uses SVD under the hood).

    Returns an integer r = number of singular values considered non-zero
    by NumPy's default tolerance.
    """
    ensure_matrix(A)
    return int(np.linalg.matrix_rank(A))


def is_full_rank(A):
    """
    Return True if rank(A) equals min(m, n).
    """
    ensure_matrix(A)
    m, n = A.shape
    return rank(A) == min(m, n)


def _basis_as_columns(basis):
    """
    Convert a set of basis vectors to a single matrix with columns as vectors.

    Accepted formats:
      - list/tuple of 1-D arrays, each shape (n,)
      - 2-D array of shape (n, k) where columns are basis vectors

    Returns: A matrix M of shape (n, k).
    """
    if isinstance(basis, (list, tuple)):
        if len(basis) == 0:
            raise ValueError("span_contains: basis cannot be empty.")
        # Ensure all are vectors of same length
        n = None
        cols = []
        for v in basis:
            ensure_vector(v)
            if n is None:
                n = v.shape[0]
            elif v.shape[0] != n:
                raise ValueError("All basis vectors must have the same length.")
            cols.append(np.asarray(v))
        M = np.stack(cols, axis=1)  # shape (n, k): columns are vectors
        return M
    else:
        M = np.asarray(basis)
        ensure_matrix(M)
        return M  # assume columns are vectors


def span_contains(v, basis, tol=None):
    """
    Return True if vector v lies in the span of the given basis (within tolerance).

    Inputs
    ------
    v : 1-D array, shape (n,)
    basis :
        - list/tuple of vectors (each shape (n,)), or
        - 2-D array of shape (n, k) with columns as basis vectors.
    tol : float or None
        If None, a default is chosen based on v's size and dtype.

    Method
    ------
    Solve the least-squares problem M x ≈ v for M whose columns are the basis.
    If the residual ||v - Mx|| is small (<= tol), we say v is in the span.
    """
    ensure_vector(v)
    M = _basis_as_columns(basis)  # shape (n, k)
    n, k = M.shape
    if v.shape[0] != n:
        raise ValueError("span_contains: dimension mismatch; v has length %d, basis vectors have length %d"
                         % (v.shape[0], n))

    # Least squares solution x to minimize ||M x - v||
    # lstsq returns x, residuals, rank, singular_values
    x, residuals, rnk, s = np.linalg.lstsq(M, v, rcond=None)

    # Compute residual explicitly (works even if residuals array is empty)
    r = v - M @ x
    res_norm = float(np.linalg.norm(r, ord=2))

    # Default tolerance: scale with vector magnitude and machine eps
    if tol is None:
        # machine eps for float64 ~ 2e-16; float32 ~ 1e-7
        eps = np.finfo(v.dtype).eps if np.issubdtype(v.dtype, np.floating) else 1e-12
        tol = 10.0 * (np.linalg.norm(v, ord=2) + 1.0) * eps * max(n, k)

    return bool(res_norm <= tol)
