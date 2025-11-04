import numpy as np
from chapters.ch06_feedforward.api import relu_forward, relu_backward

def test_relu_forward_and_backward_mask():
    Z = np.array([[-1.0, 0.0, 2.5],
                  [ 3.0,-4.0, 0.1]], dtype=np.float64)
    A, cache = relu_forward(Z)
    assert np.array_equal(A, np.array([[0.0, 0.0, 2.5],
                                       [3.0, 0.0, 0.1]]))
    upstream = np.ones_like(Z)
    dZ = relu_backward(upstream, cache)
    # grads only flow where Z>0
    expected = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 1.0]])
    assert np.array_equal(dZ, expected)

def test_relu_backward_shape_check():
    Z = np.random.randn(2, 3)
    A, cache = relu_forward(Z)
    try:
        _ = relu_backward(np.ones((3, 2)), cache)
        assert False, "expected ValueError for shape mismatch"
    except ValueError:
        pass
