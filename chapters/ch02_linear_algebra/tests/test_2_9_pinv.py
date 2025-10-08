import numpy as np
from chapters.ch02_linear_algebra.s2_9 import (
    pinv_svd, min_norm_solve
)


def test_pinv_matches_inverse_when_well_conditioned():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4)).astype(np.float64)
    A = (M.T @ M) + 0.5 * np.eye(4)  # SPD -> well conditioned, invertible

    A_inv = np.linalg.inv(A)
    A_plus = pinv_svd(A, rcond=1e-15)

    # Pseudoinverse should match the true inverse (within tiny tolerance)
    assert np.allclose(A_plus, A_inv, atol=1e-12, rtol=1e-10)

    # Moore–Penrose properties on square invertible: A A^+ = A^+ A = I
    I = np.eye(4)
    assert np.allclose(A @ A_plus, I, atol=1e-10)
    assert np.allclose(A_plus @ A, I, atol=1e-10)


def test_pinv_projector_identities_general_case():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 3)).astype(np.float64)  # tall, likely full column rank
    A_plus = pinv_svd(A)

    # Projections:
    # P_col = A A^+  (onto column space of A) -> symmetric & idempotent
    # P_row = A^+ A  (onto row space of A^T)  -> symmetric & idempotent
    P_col = A @ A_plus            # (5,5)
    P_row = A_plus @ A            # (3,3)

    assert np.allclose(P_col, P_col.T, atol=1e-10)
    assert np.allclose(P_row, P_row.T, atol=1e-10)
    assert np.allclose(P_col @ P_col, P_col, atol=1e-10)
    assert np.allclose(P_row @ P_row, P_row, atol=1e-10)


def test_min_norm_solve_vs_lstsq_full_column_rank():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 3)).astype(np.float64)  # tall (overdetermined), full col rank very likely
    x_true = rng.normal(size=(3,)).astype(np.float64)
    b = A @ x_true + 0.01 * rng.normal(size=(6,))   # small noise

    # Our min-norm solution (which for overdetermined reduces to LS solution)
    x_star = min_norm_solve(A, b, rcond=1e-12)

    # Compare with NumPy's lstsq solution (min LS error)
    x_np, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=None)

    assert x_star.shape == (3,)
    # Solutions should be very close
    assert np.allclose(x_star, x_np, atol=1e-8, rtol=1e-6)

    # Check that A x* approximates b well
    err_star = np.linalg.norm(A @ x_star - b)
    err_np = np.linalg.norm(A @ x_np - b)
    assert err_star <= err_np + 1e-12


def test_min_norm_solve_rank_deficient_underdetermined():
    # Construct rank-deficient underdetermined system (m < n, dependent columns)
    A = np.array([[1., 0., 1.],
                  [0., 1., 1.]], dtype=np.float64)  # shape (2,3), rank 2 (third col = sum of first two)
    b = np.array([1., 2.], dtype=np.float64)

    # Our minimum-norm solution
    x_star = min_norm_solve(A, b, rcond=1e-15)

    # Compare with NumPy's lstsq (which also returns the minimum-norm solution for underdetermined)
    x_np, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=None)

    assert x_star.shape == (3,)
    assert np.allclose(x_star, x_np, atol=1e-10, rtol=1e-8)

    # Minimum norm property: among all solutions to Ax=b, x* has the smallest L2 norm
    # We can verify by comparing to a different solution (if any) constructed by adding a vector in null(A)
    # Null space basis vector (since [1,1,-1] is orthogonal to rows -> A @ [1,1,-1]^T = 0).
    z = np.array([1., 1., -1.])
    # Another valid solution: x_alt = x_star + t * z
    # Choose t to make x_alt differ but still satisfy Ax=b (any t works since A z = 0)
    t = 1.234
    x_alt = x_star + t * z
    # Check they produce the same Ax
    assert np.allclose(A @ x_alt, b, atol=1e-12)
    # x_star should have smaller or equal L2 norm than x_alt
    assert np.linalg.norm(x_star) <= np.linalg.norm(x_alt) + 1e-12
