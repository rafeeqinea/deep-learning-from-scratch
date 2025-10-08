import numpy as np
from chapters.ch02_linear_algebra.s2_3 import (
    eye, projector, is_invertible, solve, maybe_inverse
)

def test_eye_and_projector():
    I3 = eye(3)
    assert I3.shape == (3, 3)
    assert np.allclose(I3, np.eye(3, dtype=np.float32))

    # Build an orthonormal basis U via QR
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 2)).astype(np.float32)  # (n=5, k=2)
    # QR decomposition -> Q has orthonormal columns
    Q, _ = np.linalg.qr(A)  # Q: (5, 2)
    P = projector(Q, tol=1e-5)  # P = Q Q^T
    # Projector should be symmetric and idempotent (approximately)
    assert P.shape == (5, 5)
    assert np.allclose(P, P.T, atol=1e-5)
    assert np.allclose(P @ P, P, atol=1e-5)

    # If U not orthonormal, projector should raise
    U_bad = A  # original A not orthonormal
    try:
        _ = projector(U_bad, tol=1e-7)
        assert False, "Expected projector to fail for non-orthonormal columns"
    except ValueError:
        pass

def test_solve_matches_inverse_on_good_matrix():
    rng = np.random.default_rng(1)
    # Make a well-conditioned square matrix
    M = rng.normal(size=(4, 4)).astype(np.float64)
    # Improve conditioning by making it symmetric positive definite
    A = (M.T @ M) + 1e-1 * np.eye(4, dtype=np.float64)

    b_vec = rng.normal(size=(4,)).astype(np.float64)
    b_mat = rng.normal(size=(4, 3)).astype(np.float64)

    x1 = solve(A, b_vec)
    x2 = solve(A, b_mat)

    # Compare with inverse-based solution (should be close)
    A_inv = np.linalg.inv(A)
    x1_ref = A_inv @ b_vec
    x2_ref = A_inv @ b_mat
    assert np.allclose(x1, x1_ref, atol=1e-10, rtol=1e-8)
    assert np.allclose(x2, x2_ref, atol=1e-10, rtol=1e-8)

def test_is_invertible_and_maybe_inverse_guardrails():
    # Singular matrix (rank deficient)
    S = np.array([[1., 2.],
                  [2., 4.]], dtype=np.float32)
    assert is_invertible(S) is False
    try:
        _ = maybe_inverse(S)
        assert False, "Expected failure: S is singular"
    except Exception:
        pass

    # Ill-conditioned matrix (huge condition number)
    A = np.array([[1.0, 0.0],
                  [0.0, 1e-14]], dtype=np.float64)
    assert is_invertible(A) is True  # full rank numerically
    # Default should refuse inversion
    try:
        _ = maybe_inverse(A, allow_ill_conditioned=False, cond_threshold=1e12)
        assert False, "Expected refusal due to ill-conditioning"
    except Exception:
        pass
    # Allow override
    A_inv = maybe_inverse(A, allow_ill_conditioned=True, cond_threshold=1e12)
    assert np.allclose(A @ A_inv, np.eye(2), atol=1e-6)
