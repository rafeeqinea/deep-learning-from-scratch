import numpy as np
from chapters.ch02_linear_algebra.s2_2 import (
    mm, mv, vv, safe_matmul, lincomb, affine_map, add_affine
)

def test_mm_mv_vv_basic():
    A = np.array([[1., 2., 3.],
                  [4., 5., 6.]], dtype=np.float32)      # (2,3)
    B = np.array([[1., 0.],
                  [0., 1.],
                  [1., 1.]], dtype=np.float32)          # (3,2)
    v = np.array([1., 2., 3.], dtype=np.float32)        # (3,)
    u = np.array([4., -1., 0.5], dtype=np.float32)      # (3,)

    # mm: (2,3) @ (3,2) -> (2,2)
    AB = mm(A, B)
    assert AB.shape == (2, 2)
    # quick numeric check
    AB_ref = A @ B
    assert np.allclose(AB, AB_ref)

    # mv: (2,3) @ (3,) -> (2,)
    Av = mv(A, v)
    assert Av.shape == (2,)
    assert np.allclose(Av, A @ v)

    # vv: (3,) · (3,) -> scalar
    dot_val = vv(v, u)
    assert np.isfinite(dot_val)
    assert np.allclose(dot_val, float(np.dot(v, u)))

def test_affine_and_lincomb():
    X = np.array([[1., 2.],
                  [3., 4.]], dtype=np.float32)          # (2,2)
    W = np.array([[2., 0.5],
                  [0., 1.0]], dtype=np.float32)         # (2,2)
    b = np.array([0.5, -1.0], dtype=np.float32)         # (2,)

    Y = affine_map(X, W, b)  # (2,2)
    assert Y.shape == (2, 2)
    assert np.allclose(Y, X @ W + b)

    Z = add_affine(Y, b)
    assert np.allclose(Z, Y + b)

    # lincomb: c1*v1 + c2*v2
    v1 = np.array([1., 0., -1.], dtype=np.float32)
    v2 = np.array([2., 2., 2.], dtype=np.float32)
    coeffs = np.array([0.5, 1.5], dtype=np.float32)
    lc = lincomb([v1, v2], coeffs)  # 0.5*v1 + 1.5*v2
    assert lc.shape == (3,)
    lc_ref = 0.5 * v1 + 1.5 * v2
    assert np.allclose(lc, lc_ref)

def test_shape_errors():
    A = np.zeros((2, 3), dtype=np.float32)
    B_bad = np.zeros((4, 2), dtype=np.float32)   # inner dims don't match
    try:
        mm(A, B_bad)
        assert False, "Expected mm to fail on incompatible shapes"
    except ValueError:
        pass

    v_bad = np.zeros((4,), dtype=np.float32)
    try:
        mv(A, v_bad)
        assert False, "Expected mv to fail on incompatible shapes"
    except ValueError:
        pass

    Y = np.zeros((5, 4), dtype=np.float32)
    b_bad = np.zeros((3,), dtype=np.float32)     # wrong bias length
    try:
        add_affine(Y, b_bad)
        assert False, "Expected add_affine to fail on bias length mismatch"
    except ValueError:
        pass

def test_safe_matmul():
    A = np.eye(3, dtype=np.float32)
    B = np.ones((3, 4), dtype=np.float32)
    C = safe_matmul(A, B)
    assert np.allclose(C, A @ B)

    # Non-2D should raise
    A3 = np.ones((2, 3, 4), dtype=np.float32)
    try:
        safe_matmul(A3, B)
        assert False, "Expected safe_matmul to reject non-2D arrays"
    except ValueError:
        pass
