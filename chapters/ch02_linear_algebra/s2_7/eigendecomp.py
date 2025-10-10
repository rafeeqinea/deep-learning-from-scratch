# Eigen-decomposition utilities for symmetric (real) matrices (NumPy only).
import numpy as np

# Reuse basic checks
from chapters.ch02_linear_algebra.s2_1 import ensure_matrix
from chapters.ch02_linear_algebra.s2_6 import is_symmetric


def _symmetrize(A):
    """
    Return (A + A^T) / 2 — the symmetric part of A.
    This removes tiny asymmetries caused by numerical noise.
    """
    return 0.5 * (A + A.T)


def eig_sym(A, tol=1e-8, allow_symmetrize=True):
    """
    Compute eigenvalues and eigenvectors of a (real) symmetric matrix A.

    Why symmetric?
      For symmetric matrices:
        - eigenvalues are real
        - eigenvectors can be chosen orthonormal
        - NumPy's eigh(eigvalsh) is stable and fast

    Parameters
    ----------
    A : 2-D array (n, n)
    tol : float
        Tolerance for symmetry check. If A is not within tol of A.T:
          - if allow_symmetrize=True, we replace A by (A + A.T)/2
          - else we raise a ValueError
    allow_symmetrize : bool
        If True (default), we gently symmetrize nearly-symmetric A.

    Returns
    -------
    w : (n,) array
        Eigenvalues in ascending order.
    V : (n, n) array
        Columns are the corresponding eigenvectors (orthonormal): V.T @ V ≈ I.
    """
    ensure_matrix(A)
    if not is_symmetric(A, tol=tol):
        if allow_symmetrize:
            A = _symmetrize(A)
        else:
            raise ValueError("eig_sym: A is not symmetric within tolerance; "
                             "either symmetrize or set allow_symmetrize=True.")

    # eigh is for Hermitian/symmetric matrices; returns real eigenvalues (ascending)
    w, V = np.linalg.eigh(A)
    # Ensure real dtype (small imaginary noise is not expected here)
    w = np.asarray(w, dtype=float)
    V = np.asarray(V, dtype=float)
    return w, V


def spectral_decomp(A, tol=1e-8):
    """
    Convenience wrapper that *always* symmetrizes first, then returns (w, V).

      A_sym = (A + A^T)/2
      A_sym = V diag(w) V^T

    This is helpful when A is supposed to be symmetric but suffers tiny
    floating-point asymmetries (common after arithmetic).
    """
    ensure_matrix(A)
    A_sym = _symmetrize(A)
    w, V = np.linalg.eigh(A_sym)
    w = np.asarray(w, dtype=float)
    V = np.asarray(V, dtype=float)
    return w, V


def is_positive_definite(A, tol=1e-12):
    """
    Return True if A is (numerically) symmetric positive definite (PD):
      all eigenvalues > tol.

    Steps:
      1) Symmetrize A to reduce tiny asymmetry: A ← (A + A^T)/2
      2) Compute eigenvalues with eigvalsh (for symmetric matrices).
      3) Check min eigenvalue > tol.

    Notes:
      - PD is stricter than PSD (which allows zeros). For PSD, check >= -tol.
      - PD implies invertible and a unique Cholesky factorization.
    """
    ensure_matrix(A)
    A = _symmetrize(A)
    w = np.linalg.eigvalsh(A)
    return bool(np.min(w) > float(tol))
