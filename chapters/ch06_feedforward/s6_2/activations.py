import numpy as np
from typing import Tuple, Dict, Any
Array = np.ndarray

def relu_forward(Z: Array) -> Tuple[Array, Dict[str, Any]]:
    """
    A = max(Z, 0)
    Returns (A, cache) where cache contains a boolean mask for backward.
    """
    Z = np.asarray(Z)
    mask = Z > 0
    A = Z * mask
    cache = {"mask": mask, "shape": Z.shape}
    return A, cache

def relu_backward(dA: Array, cache: Dict[str, Any]) -> Array:
    """
    dZ = dA * 1[Z>0], using mask from forward cache.
    """
    dA = np.asarray(dA)
    mask = cache["mask"]
    if dA.shape != cache["shape"]:
        raise ValueError("dA shape mismatch")
    return dA * mask
