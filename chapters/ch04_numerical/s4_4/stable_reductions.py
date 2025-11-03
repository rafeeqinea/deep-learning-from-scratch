import numpy as np
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, float]

def logsumexp(x: np.ndarray,
              axis: Optional[Union[int, Tuple[int, ...]]] = None,
              keepdims: bool = False) -> np.ndarray:
    """
    Numerically stable log(sum(exp(x))) along axis.
    Handles all -inf rows/columns by returning -inf for those reductions.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)  # may be -inf
    shifted = x - m
    # safe exp: set exp(NaN) to 0 where shifted is not finite
    e = np.zeros_like(shifted)
    finite_mask = np.isfinite(shifted)
    e[finite_mask] = np.exp(shifted[finite_mask])
    s = np.sum(e, axis=axis, keepdims=True)
    out = np.log(s) + m
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable log-softmax: logits - logsumexp(logits).
    Shift-invariant along the given axis.
    """
    logits = np.asarray(logits, dtype=np.float64)
    lse = logsumexp(logits, axis=axis, keepdims=True)
    return logits - lse

def softplus(x: ArrayLike) -> np.ndarray:
    """
    Stable softplus: log(1 + exp(x)) using:
      softplus(x) = log1p(exp(-|x|)) + max(x, 0)
    """
    x = np.asarray(x, dtype=np.float64)
    absx = np.abs(x)
    return np.log1p(np.exp(-absx)) + np.maximum(x, 0.0)
