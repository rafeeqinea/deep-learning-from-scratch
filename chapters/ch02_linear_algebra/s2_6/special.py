# Checks for special matrix properties (NumPy only).
import numpy as np

# Reuse shape validator from §2.1
from chapters.ch02_linear_algebra.s2_1 import ensure_matrix


def is_diagonal(A, tol=0.0):
    """
    Return True if A is (numerically) diagonal.
    That means all off-diagonal entries are zero (within tolerance).

    Strategy:
      Compare A to diag(diag(A)). If they are close (within atol=tol), return True.
    """
    ensure_matrix(A)
    D = np.diag(np.diag(A))          # keep the diagonal, zero elsewhere
    return np.allclose(A, D, atol=float(tol), rtol=0.0)


def is_symmetric(A, tol=1e-8):
    """
    Return True if A is (numerically) symmetric: A == A^T within tolerance.
    """
    ensure_matrix(A)
    return np.allclose(A, A.T, atol=float(tol), rtol=0.0)


def is_orthogonal(Q, tol=1e-8):
    """
    Return True if Q is (numerically) orthogonal (square with Q^T Q == I).
    """
    ensure_matrix(Q)
    n, m = Q.shape
    if n != m:
        return False  # must be square
    I = np.eye(n, dtype=Q.dtype)
    return np.allclose(Q.T @ Q, I, atol=float(tol), rtol=0.0)


def is_psd(A, tol=1e-10):
    """
    Return True if A is (numerically) symmetric positive semidefinite.

    Steps:
      1) Symmetry check (A ≈ A^T). If not, we can symmetrize: (A + A^T)/2.
      2) Compute eigenvalues (for symmetric matrices use eigvalsh).
      3) All eigenvalues must be >= -tol (allow tiny negatives from roundoff).

    Notes:
      - PSD includes zero eigenvalues (semidefinite).
      - For strictly positive definite (PD), you would check eigenvalues > tol.
    """
    ensure_matrix(A)

    # Symmetry check; if slightly off, symmetrize to reduce numerical noise
    if not is_symmetric(A, tol=tol):
        A = 0.5 * (A + A.T)

    # For symmetric matrices, eigvalsh is more stable and faster
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        # Fall back: use general eigvals and take real part
        w = np.linalg.eigvals(A).real

    # Allow small negative values within tolerance
    return np.min(w) >= -float(tol)
