# Invertibility checks and safe solving (NumPy only).
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix, ensure_vector
)


def is_invertible(A, tol=None):
    """
    Return True if A is square and full rank (i.e., invertible).

    We use NumPy's rank computation. If 'tol' is None, NumPy picks a default
    tolerance based on SVD and machine precision.

    Note: This is a *numerical* test; very ill-conditioned matrices may be
    technically full rank but still unsafe to invert in practice.
    """
    ensure_matrix(A)
    n, m = A.shape
    if n != m:
        return False
    rank = np.linalg.matrix_rank(A, tol=tol)
    return bool(rank == n)


def solve(A, b):
    """
    Solve the linear system A x = b using a robust solver (preferred over inverse).

    Shapes:
        A: (n, n)
        b: (n,) or (n, k)

    Returns:
        x: (n,) if b was (n,)
           (n, k) if b was (n, k)

    This will raise if A is singular or not square.
    """
    ensure_matrix(A)
    n, m = A.shape
    if n != m:
        raise ValueError("solve: A must be square, got shape %r" % (A.shape,))

    # Accept b as vector (n,) or matrix (n, k)
    if not isinstance(b, np.ndarray):
        raise TypeError("solve: b must be a NumPy array.")
    if b.ndim == 1:
        ensure_vector(b, length=n)
    elif b.ndim == 2:
        ensure_matrix(b, rows=n)  # cols = k is free
    else:
        raise ValueError("solve: b must have ndim 1 or 2, got shape %r" % (b.shape,))

    # Use NumPy's linear solver
    x = np.linalg.solve(A, b)
    return x


def maybe_inverse(A, allow_ill_conditioned=False, cond_threshold=1e12):
    """
    Attempt to compute inv(A), with guardrails.

    Rules:
      - A must be square and full rank.
      - If the condition number is very large (ill-conditioned),
        we refuse unless allow_ill_conditioned=True.

    Parameters:
      A : (n, n) array
      allow_ill_conditioned : bool
          If False (default), we raise when cond(A) > cond_threshold.
      cond_threshold : float
          Threshold above which we consider A too ill-conditioned to invert safely.

    Returns:
      A_inv : (n, n) array

    Notes:
      - In practice, prefer 'solve(A, b)' over forming inv(A).
      - Condition number cond(A) measures sensitivity; the larger it is,
        the more numerical error amplification you may see.
    """
    ensure_matrix(A)
    n, m = A.shape
    if n != m:
        raise ValueError("maybe_inverse: A must be square, got shape %r" % (A.shape,))

    # Check rank
    if not is_invertible(A):
        raise np.linalg.LinAlgError("maybe_inverse: A is singular (not invertible).")

    # Check conditioning
    cond = np.linalg.cond(A)
    if not np.isfinite(cond):
        raise np.linalg.LinAlgError("maybe_inverse: A has infinite condition number (singular).")
    if cond > float(cond_threshold) and not allow_ill_conditioned:
        raise np.linalg.LinAlgError(
            "maybe_inverse: A is ill-conditioned (cond=%.3e > %.3e). "
            "Use solve(A, b) or set allow_ill_conditioned=True if you really need inv(A)." %
            (cond, cond_threshold)
        )

    return np.linalg.inv(A)
