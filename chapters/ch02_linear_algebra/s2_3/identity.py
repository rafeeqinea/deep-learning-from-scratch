# Identity matrix and orthogonal projector (NumPy only).
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix,
)


def eye(n, dtype=np.float32):
    """
    Return the n×n identity matrix I_n.

    Example:
        eye(3) ->
        [[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]]
    """
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("eye(n): n must be a positive integer.")
    return np.eye(n_int, dtype=dtype)


def projector(U, tol=1e-6):
    """
    Return the orthogonal projector P = U U^T, assuming the columns of U are orthonormal.

    Shapes:
        U: (n, k) with k <= n and U^T U ≈ I_k

    Result:
        P: (n, n), symmetric, idempotent (P^2 ≈ P), projects onto the column space of U.

    We check that U is 2-D and that U^T U is close to the identity (within 'tol').
    """
    ensure_matrix(U)
    n, k = U.shape
    if k > n:
        raise ValueError("projector: number of columns k cannot exceed n (got n=%d, k=%d)" % (n, k))

    # Check near-orthonormality of columns: U^T U ≈ I_k
    gram = U.T @ U
    I = np.eye(k, dtype=U.dtype)
    if not np.allclose(gram, I, atol=tol, rtol=0.0):
        raise ValueError(
            "projector: columns of U are not orthonormal within tolerance. "
            "Try orthonormalizing (e.g., via QR) before calling projector."
        )

    # Orthogonal projector onto col(U)
    return U @ U.T
