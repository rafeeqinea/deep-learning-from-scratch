import numpy as np
from chapters.ch02_linear_algebra.s2_5 import (
    norm_l1, norm_l2, norm_linf, frobenius, op_norm_2
)

def test_vector_norms_basic():
    x = np.array([3.0, -4.0, 0.0], dtype=np.float64)
    # L1 = |3| + |−4| + |0| = 7
    assert np.isclose(norm_l1(x), 7.0)
    # L2 = sqrt(3^2 + (−4)^2) = 5
    assert np.isclose(norm_l2(x), 5.0)
    # Linf = max(|3|, |−4|, |0|) = 4
    assert np.isclose(norm_linf(x), 4.0)

def test_vector_norm_relations():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20).astype(np.float64)
    l1 = norm_l1(x)
    l2 = norm_l2(x)
    linf = norm_linf(x)
    n = x.shape[0]

    # Standard inequalities:
    #   ||x||_inf <= ||x||_2 <= ||x||_1
    assert linf <= l2 + 1e-12
    assert l2 <= l1 + 1e-12

    #   ||x||_1 <= sqrt(n) * ||x||_2
    assert l1 <= (n ** 0.5) * l2 + 1e-12

    #   ||x||_2 <= sqrt(n) * ||x||_inf
    assert l2 <= (n ** 0.5) * linf + 1e-12

def test_frobenius_equals_flattened_l2():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(5, 7)).astype(np.float64)
    f = frobenius(W)
    flat_l2 = np.linalg.norm(W.ravel(), ord=2)
    assert np.isclose(f, flat_l2)

def test_op_norm_bounds_Ax():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 4)).astype(np.float64)
    # Spectral norm upper-bounds ||A x||_2 / ||x||_2
    c = op_norm_2(A)
    for _ in range(10):
        x = rng.normal(size=4).astype(np.float64)
        Ax = A @ x
        lhs = np.linalg.norm(Ax, ord=2)
        rhs = c * np.linalg.norm(x, ord=2)
        assert lhs <= rhs + 1e-10  # small tolerance

def test_op_norm_diagonal_and_orthogonal_invariance():
    # Diagonal matrix: spectral norm = max |diag|
    d = np.array([3.0, -1.0, 0.5], dtype=np.float64)
    A = np.diag(d)
    assert np.isclose(op_norm_2(A), np.max(np.abs(d)))

    # Orthogonal invariance: ||Q A||_2 = ||A||_2 when Q is orthogonal (Q^T Q = I)
    rng = np.random.default_rng(3)
    M = rng.normal(size=(5, 5)).astype(np.float64)
    Q, _ = np.linalg.qr(M)   # Q is 5x5 orthogonal
    B = rng.normal(size=(5, 3)).astype(np.float64)  # rectangular
    assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-10)

    c1 = op_norm_2(B)
    c2 = op_norm_2(Q @ B)
    assert np.isclose(c1, c2, atol=1e-10)
