import numpy as np
from chapters.ch04_numerical.s4_5.gradcheck import central_diff_grad, check_grad

def test_central_diff_grad_quadratic():
    # f(x) = 0.5 * ||x||^2  -> grad = x
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3)).astype(np.float64)

    def f(z):
        return 0.5 * float(np.sum(z * z))

    num = central_diff_grad(f, x, eps=1e-6)
    assert np.allclose(num, x, rtol=1e-4, atol=1e-6)

def test_check_grad_sin_affine_sum():
    # f(x) = sum( sin(Ax + b) ), scalar output
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4)).astype(np.float64)
    b = rng.normal(size=(4,)).astype(np.float64)
    x = rng.normal(size=(4,)).astype(np.float64)

    def f(z):
        return float(np.sum(np.sin(A @ z + b)))

    # analytical grad: (A^T * cos(Ax + b))
    analytic = A.T @ np.cos(A @ x + b)
    ok, maxdiff = check_grad(f, x, analytic, rtol=1e-4, atol=1e-6, eps=1e-6)
    assert ok, f"gradcheck failed, max diff {maxdiff}"
