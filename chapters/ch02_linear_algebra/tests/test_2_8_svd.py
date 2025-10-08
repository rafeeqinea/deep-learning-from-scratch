import numpy as np
from chapters.ch02_linear_algebra.s2_8 import (
    svd_thin, cond_number, low_rank_approx
)

def test_svd_thin_reconstruction_and_orthonormality():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 4)).astype(np.float64)

    U, S, VT = svd_thin(A)

    # Shapes
    m, n = A.shape
    r = min(m, n)
    assert U.shape == (m, r)
    assert S.shape == (r,)
    assert VT.shape == (r, n)

    # Orthonormal columns/rows
    I_r = np.eye(r)
    assert np.allclose(U.T @ U, I_r, atol=1e-10)
    assert np.allclose(VT @ VT.T, I_r, atol=1e-10)

    # Reconstruction
    A_rec = (U * S) @ VT
    assert np.allclose(A_rec, A, atol=1e-10)

def test_cond_number_basic_and_singular():
    # Diagonal matrix: singular values are absolute diagonal entries (sorted)
    d = np.array([3.0, 2.0, 0.5], dtype=np.float64)
    A = np.diag(d)
    # cond = max / min = 3 / 0.5 = 6
    assert np.isclose(cond_number(A), 6.0)

    # Singular matrix -> infinite condition number
    B = np.array([[1.0, 0.0],
                  [0.0, 0.0]], dtype=np.float64)
    assert np.isinf(cond_number(B))

    # All zero matrix -> inf by convention here
    Z = np.zeros((3, 3), dtype=np.float64)
    assert np.isinf(cond_number(Z))

def test_low_rank_approx_rank_control():
    rng = np.random.default_rng(1)
    # Build a matrix with fast-decaying singular values
    U, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    V, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    s = np.array([5.0, 2.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01], dtype=np.float64)
    S = np.zeros((10, 8), dtype=np.float64)
    np.fill_diagonal(S, s)
    A = (U @ S) @ V.T   # shape (10, 8)

    # r = 0 -> zeros
    A0 = low_rank_approx(A, 0)
    assert np.allclose(A0, 0.0)

    # r >= min(m, n) -> exact reconstruction
    rmax = min(A.shape)
    A_full = low_rank_approx(A, rmax)
    assert np.allclose(A_full, A, atol=1e-10)

    # Proper low-rank approximation (e.g., r=3)
    r = 3
    A3 = low_rank_approx(A, r)
    # Error norm ~ sqrt(sum_{i>r} s_i^2) (Eckart–Young)
    err = np.linalg.norm(A - A3, ord="fro")
    err_bound = np.sqrt(np.sum(s[r:] ** 2))
    # It should be close, tolerance for numerical noise
    assert abs(err - err_bound) / (err_bound + 1e-12) < 1e-6

def test_low_rank_approx_shapes_and_values():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(7, 5)).astype(np.float64)
    A2 = low_rank_approx(A, 2)
    assert A2.shape == A.shape

    # Rank-1 approximation behaves like outer product of top singular vectors
    A1 = low_rank_approx(A, 1)
    U, S, VT = svd_thin(A)
    A1_ref = (U[:, :1] * S[:1]) @ VT[:1, :]
    assert np.allclose(A1, A1_ref, atol=1e-10)
