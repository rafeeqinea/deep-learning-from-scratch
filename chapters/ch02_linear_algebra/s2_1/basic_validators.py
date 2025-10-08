# Basic sanity checks for shapes and values (NumPy only).
import numpy as np

def check_same_shape(A, B):
    """
    Raise an error if A and B do not have exactly the same shape.
    """
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("check_same_shape expects NumPy arrays.")
    if A.shape != B.shape:
        raise ValueError("Shapes differ: %r vs %r" % (A.shape, B.shape))

def _broadcastable_right_aligned(a_shape, b_shape):
    """
    Simple check for NumPy-style broadcasting compatibility.

    Rule (right-aligned):
      For each dimension from the end, either the sizes are equal,
      or one of them is 1. Missing leading dims are treated as 1.

    Example:
      (2, 1, 3) and (3,)  -> broadcastable
      (2, 3) and (1, 3)   -> broadcastable
      (2, 3) and (2, 1)   -> broadcastable
      (2, 3) and (2, 4)   -> NOT broadcastable
    """
    a = list(a_shape)[::-1]  # reverse for right-to-left check
    b = list(b_shape)[::-1]
    max_len = max(len(a), len(b))
    for i in range(max_len):
        da = a[i] if i < len(a) else 1
        db = b[i] if i < len(b) else 1
        if not (da == db or da == 1 or db == 1):
            return False
    return True

def check_broadcastable(a_shape, b_shape):
    """
    Raise an error if a_shape and b_shape are not broadcast-compatible.
    """
    if not _broadcastable_right_aligned(tuple(a_shape), tuple(b_shape)):
        raise ValueError("Shapes not broadcastable: %r vs %r" %
                         (a_shape, b_shape))

def check_finite(X):
    """
    Raise an error if X contains NaN or +/-inf.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("check_finite expects a NumPy array.")
    if not np.all(np.isfinite(X)):
        # Show where the first few bad entries are
        bad_mask = ~np.isfinite(X)
        idx = np.argwhere(bad_mask)
        raise ValueError("Non-finite entries found at indices (first few): %r" %
                         (idx[:5].tolist(),))
