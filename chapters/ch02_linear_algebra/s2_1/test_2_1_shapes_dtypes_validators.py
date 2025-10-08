import numpy as np
from chapters.ch02_linear_algebra.s2_1 import (
    infer_shape, ensure_vector, ensure_matrix, ensure_tensor_nd,
    as_float32, as_float64, is_integer_array,
    check_same_shape, check_broadcastable, check_finite,
)

def test_shapes_and_validators():
    x = np.arange(6, dtype=np.float32)
    X = x.reshape(2, 3)
    assert infer_shape(X) == (2, 3)
    ensure_vector(x, length=6)
    ensure_matrix(X, rows=2, cols=3)
    ensure_tensor_nd(X, ndim=2)
    check_same_shape(X, np.zeros_like(X))
    check_broadcastable((2, 3), (1, 3))
    check_broadcastable((2, 1, 3), (3,))  # right-aligned broadcasting works

def test_dtypes_and_finiteness():
    X = np.array([[1, 2], [3, 4]], dtype=np.int64)
    X32 = as_float32(X)
    X64 = as_float64(X)
    assert X32.dtype == np.float32 and X64.dtype == np.float64
    assert is_integer_array(np.array([1, 2, 3], dtype=np.int32)) is True
    check_finite(np.array([1.0, 2.0, 3.0]))
    try:
        check_finite(np.array([1.0, np.inf]))
        assert False, "Expected failure on non-finite"
    except ValueError:
        pass
