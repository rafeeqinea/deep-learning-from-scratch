import numpy as np
from typing import Tuple, Dict, Any

Array = np.ndarray

def linear_forward(X: Array, W: Array, b: Array) -> Tuple[Array, Dict[str, Any]]:
    """
    Z = X @ W + b
    X: (B, D_in)
    W: (D_in, D_out)
    b: (D_out,)
    Returns (Z, cache) where cache keeps tensors for backward.
    """
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)
    if X.ndim != 2 or W.ndim != 2 or b.ndim != 1:
        raise ValueError("shapes must be X:(B,D_in) W:(D_in,D_out) b:(D_out,)")
    if X.shape[1] != W.shape[0] or W.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes for linear_forward")
    Z = X @ W + b  # row-wise bias broadcast
    cache = {"X": X, "W": W, "b": b, "Z": Z}
    return Z, cache

def linear_backward(dZ: Array, cache: Dict[str, Any]) -> Tuple[Array, Array, Array]:
    """
    Given upstream dZ = dL/dZ, compute:
      dX = dZ @ W^T
      dW = X^T @ dZ
      db = sum over batch of dZ
    Shapes match forward.
    """
    X = cache["X"]
    W = cache["W"]
    b = cache["b"]
    dZ = np.asarray(dZ)
    if dZ.shape != cache["Z"].shape:
        raise ValueError("dZ has wrong shape")
    dX = dZ @ W.T
    dW = X.T @ dZ
    db = np.sum(dZ, axis=0)
    return dX, dW, db
