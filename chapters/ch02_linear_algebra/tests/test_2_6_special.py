import numpy as np
from chapters.ch02_linear_algebra.s2_6 import (
    is_diagonal, is_symmetric, is_orthogonal, is_psd
)

def test_is_diagonal_basic():
    D = np.diag([1.0, -2.0, 3.0]).astype(np.float64)
    assert is_diagonal(D)
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.1],   # off-diagonal non-zero
                  [0.0, 0.0, 3.0]])
    assert is_diagonal(A) is False
    # With a tolerance we can treat tiny off-diagonals as zero
    B = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 1e-12],
                  [0.0, 0.0, 3.0]])
    assert is_diagonal(B, tol=1e-10)

def test_is_symmetric_basic():
    S = np.array([[1.0, 2.0],
                  [2.0, 3.0]], dtype=np.float64)
    assert is_symmetric(S)
    NS = np.array([[1.0, 2.0],
                   [2.0, 3.0 + 1e-6]], dtype=np.float64)
    assert is_symmetric(NS) is False
    # Tiny asymmetry can be tolerated with a larger tol
    assert is_symmetric(NS, tol=1e-5)

def test_is_orthogonal_qr():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5)).astype(np.float64)
    Q, _ = np.linalg.qr(M)             # Q orthogonal (Q^T Q = I)
    assert is_orthogonal(Q)
    # Scaling breaks orthogonality
    assert is_orthogonal(2.0 * Q) is False
    # Non-square should be False
    Qrect = Q[:, :3]
    assert is_orthogonal(Qrect) is False

def test_is_psd_construction_and_tolerance():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(6, 4)).astype(np.float64)
    # A = B B^T is symmetric positive semidefinite
    A = B @ B.T
    assert is_psd(A)

    # Slightly perturb to create a tiny negative eigenvalue
    # (still should be treated as PSD under a reasonable tolerance)
    eps = 1e-12
    A_perturb = A.copy()
    A_perturb[0, 1] += eps
    A_perturb[1, 0] += eps
    assert is_psd(A_perturb, tol=1e-10)

    # Non-PSD example: make a symmetric matrix with a clearly negative eigenvalue
    C = np.array([[1.0, 0.0],
                  [0.0, -0.5]], dtype=np.float64)  # one negative eigenvalue
    assert is_psd(C) is False
