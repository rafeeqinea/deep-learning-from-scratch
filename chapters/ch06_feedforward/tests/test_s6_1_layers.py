import numpy as np
from chapters.ch06_feedforward.api import linear_forward, linear_backward
from chapters.ch04_numerical.api import central_diff_grad

def test_linear_forward_shapes_and_values():
    X = np.array([[1.0, 2.0],
                  [0.5, -1.0]], dtype=np.float64)        # (B=2, Din=2)
    W = np.array([[ 2.0, -1.0],
                  [-3.0,  4.0]], dtype=np.float64)       # (2,2)
    b = np.array([0.5, -0.5], dtype=np.float64)          # (2,)
    Z, cache = linear_forward(X, W, b)
    expected = X @ W + b
    assert np.allclose(Z, expected)
    assert cache["X"].shape == X.shape and cache["W"].shape == W.shape

def test_linear_backward_matches_numeric_grads():
    rng = np.random.default_rng(0)
    B, Din, Dout = 3, 4, 2
    X = rng.normal(size=(B, Din))
    W = rng.normal(size=(Din, Dout))
    b = rng.normal(size=(Dout,))

    # Loss: 0.5 * ||Z||^2  => dZ = Z
    def f_X(Xvar):
        Xvar = Xvar.reshape(B, Din)
        Z = Xvar @ W + b
        return 0.5 * float(np.sum(Z * Z))

    def f_W(Wvar):
        Wvar = Wvar.reshape(Din, Dout)
        Z = X @ Wvar + b
        return 0.5 * float(np.sum(Z * Z))

    def f_b(bvar):
        bvar = bvar.reshape(Dout,)
        Z = X @ W + bvar
        return 0.5 * float(np.sum(Z * Z))

    Z, cache = linear_forward(X, W, b)
    dZ = Z.copy()
    dX, dW, db = linear_backward(dZ, cache)

    num_dX = central_diff_grad(lambda xv: f_X(xv), X.copy(), eps=1e-6)
    num_dW = central_diff_grad(lambda wv: f_W(wv), W.copy(), eps=1e-6)
    num_db = central_diff_grad(lambda bv: f_b(bv), b.copy(), eps=1e-6)

    assert np.allclose(dX, num_dX, rtol=1e-5, atol=1e-7)
    assert np.allclose(dW, num_dW, rtol=1e-5, atol=1e-7)
    assert np.allclose(db, num_db, rtol=1e-5, atol=1e-7)
