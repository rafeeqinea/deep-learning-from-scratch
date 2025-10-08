# SVD utilities (NumPy only), written simply with clear checks.
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import ensure_matrix


def svd_thin(A):
    """
    Thin SVD of A:
        A = U @ np.diag(S) @ VT

    Shapes (with full_matrices=False):
        A:  (m, n)
        U:  (m, r)
        S:  (r,)
        VT: (r, n)
      where r = min(m, n)

    Notes:
      - Columns of U are orthonormal: U.T @ U = I_r
      - Rows of VT are orthonormal:   VT @ VT.T = I_r
      - S is sorted from largest to smallest
    """
    ensure_matrix(A)
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # Keep numeric dtype of A (usually float64/float32)
    U = U.astype(A.dtype, copy=False)
    S = S.astype(A.dtype, copy=False)
    VT = VT.astype(A.dtype, copy=False)
    return U, S, VT


def cond_number(A, eps=1e-15):
    """
    Spectral condition number of A (2-norm):
        cond(A) = sigma_max(A) / sigma_min(A)

    - If the smallest singular value is ~0 (<= eps * sigma_max), we return np.inf.
    - For the all-zero matrix, sigma_max = 0 and sigma_min = 0 -> cond = np.inf by convention.
    """
    ensure_matrix(A)
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return np.inf
    s_max = float(s[0])
    s_min = float(s[-1])
    # If s_max == 0, matrix is all zeros -> treat as infinite condition number
    if s_max == 0.0:
        return np.inf
    # Relative threshold to decide "numerical zero" for s_min
    if s_min <= eps * s_max:
        return np.inf
    return s_max / s_min


def low_rank_approx(A, r):
    """
    Best rank-r approximation to A in Frobenius norm (Eckart–Young):
        A_r = U_r @ diag(S_r) @ V_r^T
      where U_r, S_r, V_r are the top-r singular triplets.

    Inputs:
        A : (m, n) array
        r : int, desired rank, 0 <= r <= min(m, n)

    Returns:
        A_r : (m, n) array

    Notes:
      - If r == 0, returns the zero matrix (same shape as A).
      - If r >= min(m, n), returns A (exact reconstruction with thin SVD).
    """
    ensure_matrix(A)
    m, n = A.shape
    rmax = min(m, n)
    r = int(r)
    if r < 0:
        raise ValueError("low_rank_approx: r must be >= 0")
    if r == 0:
        return np.zeros_like(A)
    if r >= rmax:
        # The thin SVD reconstructs A exactly
        U, S, VT = svd_thin(A)
        return (U * S) @ VT  # still fine, but equals A numerically

    U, S, VT = svd_thin(A)
    U_r = U[:, :r]          # (m, r)
    S_r = S[:r]             # (r,)
    VT_r = VT[:r, :]        # (r, n)
    # U_r @ diag(S_r) @ VT_r  -> use broadcasting for efficiency: (U_r * S_r) @ VT_r
    A_r = (U_r * S_r) @ VT_r
    return A_r
