# Core matrix/vector multiplications with simple checks (NumPy only).
import numpy as np

# We reuse the basic checks from §2.1
from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix, ensure_vector, check_same_shape
)


def mm(A, B):
    """
    Matrix × Matrix product: (m, n) @ (n, p) -> (m, p)

    We enforce that A and B are 2-D matrices and inner dimensions match.
    """
    ensure_matrix(A)
    ensure_matrix(B)
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError("Matrix shape mismatch for mm: %r @ %r (n != n2)"
                         % (A.shape, B.shape))
    # Use the @ operator (same as np.matmul for 2-D)
    return A @ B


def mv(A, v):
    """
    Matrix × Vector product: (m, n) @ (n,) -> (m,)

    Note: v must be a 1-D array of length n. If you have (n, 1),
    convert it with v = v.reshape(-1) before calling mv.
    """
    ensure_matrix(A)
    ensure_vector(v)
    m, n = A.shape
    if v.shape[0] != n:
        raise ValueError("Matrix/Vector shape mismatch for mv: %r @ %r"
                         % (A.shape, v.shape))
    return A @ v


def vv(a, b):
    """
    Vector · Vector (dot product): (n,) · (n,) -> scalar (float)

    Both inputs must be 1-D arrays of the same length.
    """
    ensure_vector(a)
    ensure_vector(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vector length mismatch for vv: %r vs %r"
                         % (a.shape, b.shape))
    # np.dot on 1-D arrays returns a scalar
    return float(np.dot(a, b))


def safe_matmul(A, B):
    """
    A guarded matrix multiplication that gives clearer error messages.

    Rules:
      - If both A and B are 2-D, behave like mm(A, B).
      - If shapes are not 2-D matrices, raise a friendly error.
    """
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("safe_matmul expects NumPy arrays.")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("safe_matmul supports only 2-D matrices; got %r and %r"
                         % (A.shape, B.shape))
    return mm(A, B)
