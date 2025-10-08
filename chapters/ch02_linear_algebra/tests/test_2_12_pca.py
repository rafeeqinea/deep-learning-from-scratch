import numpy as np
from chapters.ch02_linear_algebra.s2_12 import (
    center, pca_svd, explained_variance, project_to_k, reconstruct_from_k
)

def test_center_shapes_and_means():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4)).astype(np.float64)
    Xc, mu = center(X)
    assert Xc.shape == X.shape
    assert mu.shape == (4,)
    # Centering: column means should be ~0 (up to numerical noise)
    assert np.allclose(Xc.mean(axis=0), np.zeros(4), atol=1e-12)

def test_pca_svd_reconstruction_and_scores():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 5)).astype(np.float64)
    Xc, mu = center(X)
    U, S, VT = pca_svd(Xc)

    # Thin SVD reconstruction
    Xc_rec = (U * S) @ VT
    assert np.allclose(Xc_rec, Xc, atol=1e-10)

    # Scores Z = U * S (coordinates in full PC space)
    Z_full = U * S
    # Project to k via VT and compare to taking first k columns of Z_full
    for k in [0, 1, 3, min(X.shape)]:
        Z_k = project_to_k(Xc, VT, k)        # (m, k)
        assert Z_k.shape == (X.shape[0], k)
        Z_k_alt = Z_full[:, :k]              # should span the same subspace
        # Not necessarily equal element-wise (depends on SVD sign choices),
        # but norms of columns should match singular values:
        if k > 0:
            col_norms = np.linalg.norm(Z_k, axis=0)
            assert np.allclose(col_norms, S[:k], atol=1e-10)

def test_explained_variance_and_cumulative():
    # Make a matrix with known singular values using SVD construction
    rng = np.random.default_rng(2)
    m, n = 15, 6
    U, _ = np.linalg.qr(rng.normal(size=(m, m)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    S_true = np.array([5.0, 3.0, 1.0, 0.5, 0.2, 0.1], dtype=np.float64)
    # Build centered-like matrix with those singulars
    S_mat = np.zeros((m, n), dtype=np.float64)
    np.fill_diagonal(S_mat, S_true)
    Xc = (U[:, :n] @ S_mat) @ V.T

    # SVD should recover S_true (up to tiny noise)
    Ux, S, VTx = pca_svd(Xc)
    assert np.allclose(S, S_true, atol=1e-10)

    var_ratio, cum_ratio = explained_variance(S)
    assert var_ratio.shape == (n,)
    assert cum_ratio.shape == (n,)
    # Ratios sum to 1, cumulative ends at 1
    assert np.isclose(var_ratio.sum(), 1.0)
    assert np.isclose(cum_ratio[-1], 1.0)
    # Ratios follow square of singulars
    lam = S ** 2
    assert np.allclose(var_ratio, lam / lam.sum())

def test_projection_and_reconstruction_error_matches_tail_energy():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 8)).astype(np.float64)
    Xc, mu = center(X)
    U, S, VT = pca_svd(Xc)

    for k in [0, 1, 3, 8]:
        Z_k = project_to_k(Xc, VT, k)
        Xc_hat = reconstruct_from_k(Z_k, VT, k)
        # Frobenius reconstruction error should match tail singular values
        err = np.linalg.norm(Xc - Xc_hat, ord="fro")
        tail = S[k:]
        tail_energy = np.sqrt(np.sum(tail ** 2))
        assert np.isclose(err, tail_energy, atol=1e-10)

    # Add the mean back to get full-space reconstructions if needed:
    k = 3
    Z_k = project_to_k(Xc, VT, k)
    Xc_hat = reconstruct_from_k(Z_k, VT, k)
    X_hat = Xc_hat + mu  # back to original (un-centered) space
    assert X_hat.shape == X.shape
