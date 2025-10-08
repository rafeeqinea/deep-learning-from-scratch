# Basic dtype utilities (NumPy only).
import numpy as np

def as_float32(X):
    """
    Return X as a float32 NumPy array. Copies only if needed.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("as_float32 expects a NumPy array.")
    return np.asarray(X, dtype=np.float32)

def as_float64(X):
    """
    Return X as a float64 NumPy array. Copies only if needed.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("as_float64 expects a NumPy array.")
    return np.asarray(X, dtype=np.float64)

def is_integer_array(y):
    """
    Return True if y is a NumPy array with an integer dtype (signed or unsigned).
    """
    if not isinstance(y, np.ndarray):
        return False
    return np.issubdtype(y.dtype, np.integer)
