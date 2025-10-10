import numpy as np
from chapters.ch02_linear_algebra.s2_4 import (
    rank, is_full_rank, span_contains, orthonormal_basis
)

def test_rank_and_full_rank():
    # Rank-1 matrix
    A = np.array([[1., 2.],
                  [2., 4.]], dtype=np.float64)
    assert rank(A) == 1
    assert is_full_rank(A) is False

    # Full-rank square matrix
    B = np.array([[1., 0., 0.],
                  [0., 2., 0.],
                  [0., 0., 3.]], dtype=np.float64)
    assert rank(B) == 3
    assert is_full_rank(B) is True

    # Tall full-column-rank matrix
    C = np.array([[1., 0.],
                  [0., 1.],
                  [1., 1.]], dtype=np.float64)
    assert rank(C) == 2
    assert is_full_rank(C) is True  # min(m, n) = 2

def test_span_contains_list_and_matrix_basis():
    e1 = np.array([1., 0., 0.], dtype=np.float64)
    e2 = np.array([0., 1., 0.], dtype=np.float64)
    e3 = np.array([0., 0., 1.], dtype=np.float64)

    # v is in span{e1, e2}
    v = e1 + 2.0 * e2
    assert span_contains(v, [e1, e2]) is True

    # Same with a matrix whose columns are e1 and e2
    E12 = np.stack([e1, e2], axis=1)  # shape (3, 2)
    assert span_contains(v, E12) is True

    # Vector not in span{e1} (has e2 component)
    assert span_contains(v, [e1]) is False

    # Tiny numerical tolerance should still work for near-span vectors
    v_perturb = v + 1e-12 * e3
    assert span_contains(v_perturb, [e1, e2], tol=1e-10) is True

def test_orthonormal_basis_matches_column_space():
    rng = np.random.default_rng(42)
    # Random tall matrix (likely full column rank)
    A = rng.normal(size=(5, 3)).astype(np.float64)

    # Make one column dependent to create rank 2
    A[:, 2] = A[:, 0] + A[:, 1]  # now col3 is in span of col1, col2
    r = rank(A)
    assert r == 2

    # SVD-based orthonormal basis
    Q = orthonormal_basis(A, method="svd")
    # Q should have orthonormal columns: Q^T Q ≈ I
    I = Q.T @ Q
    assert np.allclose(I, np.eye(r), atol=1e-10)

    # Projection of A onto span(Q) should recover A (up to numerical noise)
    A_proj = Q @ (Q.T @ A)
    assert np.allclose(A_proj, A, atol=1e-10)

    # QR path should also work (falls back to SVD since rank-deficient)
    Q_qr = orthonormal_basis(A, method="qr")
    assert np.allclose(Q_qr.T @ Q_qr, np.eye(r), atol=1e-10)
    A_proj_qr = Q_qr @ (Q_qr.T @ A)
    assert np.allclose(A_proj_qr, A, atol=1e-10)
