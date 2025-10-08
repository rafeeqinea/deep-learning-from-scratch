import numpy as np

def infer_shape(x):
    """
    Return the shape of a NumPy array as a tuple, e.g. (rows, cols).

    Example:
        x = np.zeros((2, 3))
        infer_shape(x)  ->  (2, 3)
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("infer_shape expects a NumPy array (np.ndarray).")
    # tuple() makes sure we return a plain tuple of ints
    return tuple(int(d) for d in x.shape)


def ensure_vector(x, length=None):
    """
    Check that x is a 1-D array (a vector). Optionally check its length.
    Raises an error if the check fails.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("ensure_vector expects a NumPy array.")
    if x.ndim != 1:
        raise ValueError("Expected a 1-D vector, got shape %r" % (x.shape,))
    if length is not None and x.shape[0] != length:
        raise ValueError("Vector length mismatch: expected %d, got %d" %
                         (length, x.shape[0]))


def ensure_matrix(X, rows=None, cols=None):
    """
    Check that X is a 2-D array (a matrix). Optionally check its (rows, cols).
    Raises an error if the check fails.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("ensure_matrix expects a NumPy array.")
    if X.ndim != 2:
        raise ValueError("Expected a 2-D matrix, got shape %r" % (X.shape,))
    r, c = X.shape
    if rows is not None and r != rows:
        raise ValueError("Row mismatch: expected %d, got %d" % (rows, r))
    if cols is not None and c != cols:
        raise ValueError("Col mismatch: expected %d, got %d" % (cols, c))


def ensure_tensor_nd(X, ndim):
    """
    Check that X has exactly `ndim` dimensions.
    Raises an error if the check fails.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("ensure_tensor_nd expects a NumPy array.")
    if X.ndim != int(ndim):
        raise ValueError("Expected ndim=%d, got %d for shape %r" %
                         (int(ndim), X.ndim, X.shape))
