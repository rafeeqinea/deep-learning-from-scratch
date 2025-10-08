# Construct an orthonormal basis for the column space (NumPy only).
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix,
)
from chapters.ch02_linear_algebra.s2_4.span_rank import rank as matrix_rank


def orthonormal_basis(A, method="svd", tol=None):
    """
    Return an orthonormal basis Q for the column space of A.

    Inputs
    ------
    A : 2-D array of shape (m, n)
    method : "svd" or "qr"
        - "svd": robust for rank-deficient A (recommended)
        - "qr" : fast when A has full column rank; falls back to SVD if not
    tol : float or None
        Rank tolerance. If None, we use NumPy defaults.

    Output
    ------
    Q : 2-D array of shape (m, r) with orthonormal columns (Q^T Q = I_r),
        where r = rank(A). The columns of Q span the same column space as A.

    Notes
    -----
    - For "svd", we compute SVD and take the first r left-singular vectors.
    - For "qr", if A is rank-deficient we fall back to SVD (NumPy's QR
      without pivoting does not reliably extract independent columns).
    """
    ensure_matrix(A)
    m, n = A.shape
    r = matrix_rank(A) if tol is None else int(np.linalg.matrix_rank(A, tol=tol))

    if r == 0:
        # No nonzero columns; return an empty (m, 0) matrix
        return np.zeros((m, 0), dtype=A.dtype)

    if method.lower() == "svd":
        # Thin SVD
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        Q = U[:, :r]  # first r left-singular vectors span col(A)
        # Normalize columns for good measure (should already be orthonormal)
        # but keep as-is; SVD returns orthonormal U.
        return Q.astype(A.dtype, copy=False)

    elif method.lower() == "qr":
        # QR is fine when columns are independent (full column rank).
        # If not full column rank, fall back to SVD for stability.
        if r < min(m, n):
            # rank-deficient: use SVD instead
            U, S, VT = np.linalg.svd(A, full_matrices=False)
            Q = U[:, :r]
            return Q.astype(A.dtype, copy=False)
        # Reduced QR: Q has shape (m, n), R is (n, n)
        Q, R = np.linalg.qr(A, mode="reduced")
        # Take only first r columns (here r == n if full column rank)
        Qr = Q[:, :r]
        return Qr.astype(A.dtype, copy=False)

    else:
        raise ValueError('orthonormal_basis: method must be "svd" or "qr".')
