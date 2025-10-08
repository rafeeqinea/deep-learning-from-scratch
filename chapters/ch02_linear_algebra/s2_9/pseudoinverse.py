# Moore–Penrose pseudoinverse and minimum–norm solver (NumPy only).
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix, ensure_vector
)


def pinv_svd(A, rcond=1e-15):
    """
    Compute the Moore–Penrose pseudoinverse A^+ using SVD.

    If A = U diag(s) V^T  (thin SVD), then
        A^+ = V diag(s^+) U^T
      where s^+_i = 0 if s_i <= rcond * max(s), else 1/s_i.

    Parameters
    ----------
    A : 2-D array, shape (m, n)
    rcond : float
        Relative cutoff for small singular values. Values below rcond * max(s)
        are treated as zero (helps numerical stability).

    Returns
    -------
    A_plus : 2-D array, shape (n, m)
    """
    ensure_matrix(A)

    # Thin SVD: A = U @ diag(S) @ VT
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # Decide which singular values to keep (relative to the largest)
    s_max = S[0] if S.size > 0 else 0.0
    cutoff = float(rcond) * float(s_max)
    S_inv = np.zeros_like(S)
    for i in range(S.size):
        if S[i] > cutoff:
            S_inv[i] = 1.0 / S[i]
        else:
            S_inv[i] = 0.0

    # A^+ = V @ diag(S_inv) @ U^T
    # Use broadcasting: (VT.T * S_inv) is V @ diag(S_inv)
    A_plus = (VT.T * S_inv) @ U.T

    # Keep dtype consistent with A (usually float64/32)
    return A_plus.astype(A.dtype, copy=False)


def min_norm_solve(A, b, rcond=1e-15):
    """
    Solve A x ≈ b in the least-squares sense and return the *minimum–norm* solution:
        x* = A^+ b

    Works for:
      - overdetermined systems (m > n)
      - underdetermined systems (m < n)
      - rank-deficient systems

    Parameters
    ----------
    A : 2-D array, shape (m, n)
    b : 1-D array (m,) or 2-D array (m, k)
    rcond : float
        Relative cutoff used by pinv_svd.

    Returns
    -------
    x : 1-D array (n,) if b was (m,)
        2-D array (n, k) if b was (m, k)
    """
    ensure_matrix(A)
    if not isinstance(b, np.ndarray):
        raise TypeError("min_norm_solve: b must be a NumPy array.")

    m, n = A.shape
    if b.ndim == 1:
        ensure_vector(b, length=m)
    elif b.ndim == 2:
        # allow (m,k)
        if b.shape[0] != m:
            raise ValueError("min_norm_solve: b must have shape (m,) or (m,k); got %r" % (b.shape,))
    else:
        raise ValueError("min_norm_solve: b must be 1-D or 2-D; got ndim=%d" % b.ndim)

    A_plus = pinv_svd(A, rcond=rcond)
    x = A_plus @ b
    return x
