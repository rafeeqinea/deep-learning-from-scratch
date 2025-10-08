# Linear combinations and simple affine maps (NumPy only).
import numpy as np

from chapters.ch02_linear_algebra.s2_1 import (
    ensure_matrix, ensure_vector, check_broadcastable
)


def lincomb(vectors, coeffs):
    """
    Linear combination of k vectors in R^n:
        result = sum_{i=1..k} coeffs[i] * vectors[i]

    Inputs:
      - vectors: list of 1-D arrays, each shape (n,)
      - coeffs:  1-D array of shape (k,)

    Output:
      - result: 1-D array of shape (n,)
    """
    if not isinstance(vectors, (list, tuple)) or len(vectors) == 0:
        raise ValueError("lincomb expects a non-empty list of vectors.")

    # Check all vectors are 1-D and same length
    n = None
    for i, v in enumerate(vectors):
        ensure_vector(v)
        if n is None:
            n = v.shape[0]
        elif v.shape[0] != n:
            raise ValueError("All vectors must have the same length; "
                             "got %d and %d" % (n, v.shape[0]))

    ensure_vector(coeffs)
    k = coeffs.shape[0]
    if k != len(vectors):
        raise ValueError("Number of coeffs (%d) must match number of vectors (%d)"
                         % (k, len(vectors)))

    # Stack vectors into a (k, n) array
    M = np.stack(vectors, axis=0)  # shape (k, n)
    # Multiply each row by its coefficient and sum over rows
    # (coeffs.reshape(k,1) broadcasts across columns)
    result = (coeffs.reshape(k, 1) * M).sum(axis=0)  # shape (n,)
    return result


def affine_map(X, W, b):
    """
    Affine map:
        Y = X @ W + b

    Shapes:
        X: (m, n)
        W: (n, p)
        b: (p,)

    Output:
        Y: (m, p)

    We check shapes and rely on NumPy's row-wise broadcasting for + b.
    """
    ensure_matrix(X)
    ensure_matrix(W)
    ensure_vector(b)
    m, n = X.shape
    n2, p = W.shape
    if n != n2:
        raise ValueError("X @ W shape mismatch: %r @ %r" % (X.shape, W.shape))
    if b.shape[0] != p:
        raise ValueError("Bias length %d must match output dim p=%d" % (b.shape[0], p))

    # Check broadcasting (m, p) + (p,)
    check_broadcastable((m, p), (p,))
    return X @ W + b


def add_affine(Y, b):
    """
    Add a bias vector to each row of Y:
        Z = Y + b

    Shapes:
        Y: (m, p)
        b: (p,)
    """
    ensure_matrix(Y)
    ensure_vector(b)
    m, p = Y.shape
    if b.shape[0] != p:
        raise ValueError("Bias length %d must match number of columns p=%d"
                         % (b.shape[0], p))
    check_broadcastable((m, p), (p,))
    return Y + b
