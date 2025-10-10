import numpy as np
from chapters.ch02_linear_algebra.s2_7 import (
    eig_sym, spectral_decomp, is_positive_definite
)

def test_eig_sym_spd_matrix():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5)).astype(np.float64)
    A = M.T @ M + 1e-1 * np.eye(5)   # symmetric positive definite (SPD)

    w, V = eig_sym(A)  # should not need symmetrization
    # Shapes
    assert w.shape == (5,)
    assert V.shape == (5, 5)
    # Orthonormal eigenvectors: V^T V ≈ I
    assert np.allclose(V.T @ V, np.eye(5), atol=1e-10)
    # Reconstruction: A ≈ V diag(w) V^T
    A_rec = (V * w) @ V.T  # scale columns of V → V @ diag(w) @ V.T
    assert np.allclose(A_rec, A, atol=1e-10)
    # Eigenvalues are >= 0.1 because of the +0.1*I shift
    assert np.min(w) >= 0.1 - 1e-12

def test_spectral_decomp_symmetrizes_small_asymmetry():
    # Make a nearly-symmetric matrix (tiny asymmetry)
    A = np.array([[2.0, 1.0],
                  [1.0 + 1e-12, 3.0]], dtype=np.float64)
    # spectral_decomp symmetrizes first
    w, V = spectral_decomp(A)
    A_sym = 0.5 * (A + A.T)
    A_rec = (V * w) @ V.T
    assert np.allclose(A_rec, A_sym, atol=1e-12)

def test_is_positive_definite_and_negatives():
    # SPD example
    B = np.array([[2.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    assert is_positive_definite(B, tol=1e-12) is True

    # PSD but not PD (zero eigenvalue)
    C = np.array([[1.0, 0.0],
                  [0.0, 0.0]], dtype=np.float64)
    assert is_positive_definite(C, tol=1e-12) is False

    # Not PSD (one negative eigenvalue)
    D = np.array([[1.0, 0.0],
                  [0.0, -0.1]], dtype=np.float64)
    assert is_positive_definite(D, tol=1e-12) is False

def test_eig_sym_raises_on_large_asymmetry_without_symmetrize():
    # Deliberately not symmetric by a large margin
    A = np.array([[0.0, 2.0],
                  [3.0, 0.0]], dtype=np.float64)  # A != A^T (off by 1)
    try:
        eig_sym(A, tol=1e-8, allow_symmetrize=False)
        assert False, "Expected eig_sym to raise on non-symmetric input"
    except ValueError:
        pass
    # With symmetrization allowed, it should work (on the symmetric part)
    w, V = eig_sym(A, tol=1e-8, allow_symmetrize=True)
    A_sym = 0.5 * (A + A.T)
    A_rec = (V * w) @ V.T
    assert np.allclose(A_rec, A_sym, atol=1e-12)
