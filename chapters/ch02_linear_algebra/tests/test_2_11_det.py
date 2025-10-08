import numpy as np
from chapters.ch02_linear_algebra.s2_11 import (
    det, logdet_safe, sign_logabsdet, volume_scale
)

def test_det_identity_and_multiplicative():
    I = np.eye(4, dtype=np.float64)
    assert np.isclose(det(I), 1.0)

    # Multiplicative: det(AB) = det(A) * det(B) for same-size square matrices
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3)).astype(np.float64)
    B = rng.normal(size=(3, 3)).astype(np.float64)
    # Use slightly "nicer" matrices by adding a small diagonal shift to avoid near-singularity
    A = A + 0.1 * np.eye(3)
    B = B + 0.1 * np.eye(3)

    left = det(A @ B)
    right = det(A) * det(B)
    assert np.allclose(left, right, atol=1e-10, rtol=1e-10)

def test_logdet_safe_spd():
    # Make SPD: M.T @ M + alpha*I
    rng = np.random.default_rng(1)
    M = rng.normal(size=(5, 5)).astype(np.float64)
    A = (M.T @ M) + 1e-1 * np.eye(5)
    # Positive definite => det>0 so logdet finite
    logd = logdet_safe(A, assume_spd=True)
    # Compare to np.log(det(A)) (should be close; slogdet is more stable)
    assert np.allclose(logd, np.log(det(A)), atol=1e-10, rtol=1e-10)

def test_sign_logabsdet_and_volume_scale():
    # Negative determinant case (e.g., reflection matrix)
    R = np.diag([-1.0, 1.0, 2.0]).astype(np.float64)  # det = -2
    s, logabs = sign_logabsdet(R)
    assert s == -1.0
    assert np.isclose(np.exp(logabs), abs(det(R)))
    # volume_scale returns |det|
    assert np.isclose(volume_scale(R), abs(det(R)))

def test_logdet_safe_raises_on_nonpositive_det():
    # Singular matrix -> sign=0, log|det|=-inf
    S = np.array([[1.0, 2.0],
                  [2.0, 4.0]], dtype=np.float64)  # rank 1 -> det 0
    try:
        _ = logdet_safe(S)
        assert False, "Expected logdet_safe to raise on singular matrix"
    except ValueError:
        pass

    # Negative determinant -> should raise unless assume_spd=True (which will also raise)
    N = np.diag([-2.0, 3.0]).astype(np.float64)  # det < 0
    try:
        _ = logdet_safe(N)
        assert False, "Expected logdet_safe to raise on negative determinant"
    except ValueError:
        pass

    # Even with assume_spd=True it should raise, because matrix is not PD
    try:
        _ = logdet_safe(N, assume_spd=True)
        assert False, "Expected logdet_safe to raise with assume_spd=True on non-PD matrix"
    except ValueError:
        pass
