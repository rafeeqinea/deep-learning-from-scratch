# Vector and matrix norms (NumPy only), written simply with clear checks.
import numpy as np

# Reuse shape validators from §2.1
from chapters.ch02_linear_algebra.s2_1 import (
    ensure_vector, ensure_matrix
)


# -----------------------
# Vector norms (1-D only)
# -----------------------

def norm_l1(x):
    """
    Vector L1 norm: sum of absolute values.
        ||x||_1 = sum_i |x_i|
    """
    ensure_vector(x)
    return float(np.sum(np.abs(x)))


def norm_l2(x):
    """
    Vector L2 norm (Euclidean length).
        ||x||_2 = sqrt(sum_i x_i^2)
    """
    ensure_vector(x)
    # np.linalg.norm with ord=2 on a 1-D array returns the Euclidean norm
    return float(np.linalg.norm(x, ord=2))


def norm_linf(x):
    """
    Vector L-infinity norm (max absolute entry).
        ||x||_inf = max_i |x_i|
    """
    ensure_vector(x)
    return float(np.max(np.abs(x)))


# ------------------------
# Matrix norms (2-D only)
# ------------------------

def frobenius(W):
    """
    Matrix Frobenius norm:
        ||W||_F = sqrt(sum_{i,j} W_{ij}^2)
    This equals the L2 norm of W flattened into a vector.
    """
    ensure_matrix(W)
    return float(np.linalg.norm(W, ord="fro"))


def op_norm_2(A):
    """
    Matrix operator 2-norm (spectral norm), i.e., the largest singular value.
        ||A||_2 = sigma_max(A)

    This is the tightest constant C such that:
        ||A @ x||_2 <= C * ||x||_2   for all vectors x.

    Implementation detail:
      We compute singular values via SVD and take the largest one.
      NumPy returns them sorted from largest to smallest.
    """
    ensure_matrix(A)
    # compute_uv=False gives only singular values (fast & simple)
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    # s[0] is the largest singular value
    return float(s[0]) if s.size > 0 else 0.0
