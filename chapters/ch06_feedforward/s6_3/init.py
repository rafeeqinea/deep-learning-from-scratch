import numpy as np
from typing import Tuple, Optional

def _compute_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    fan_in, fan_out for dense weight matrix (D_in, D_out).
    """
    if len(shape) != 2:
        raise ValueError("glorot_uniform expects 2D weight shape (fan_in, fan_out)")
    return int(shape[0]), int(shape[1])

def glorot_uniform(shape: Tuple[int, int],
                   rng: Optional[np.random.Generator] = None,
                   gain: float = 1.0,
                   dtype=np.float32) -> np.ndarray:
    """
    Xavier/Glorot uniform init.
    U[-a, a], a = gain * sqrt(6 / (fan_in + fan_out))
    """
    fan_in, fan_out = _compute_fans(shape)
    a = gain * np.sqrt(6.0 / float(fan_in + fan_out))
    g = rng if rng is not None else np.random.default_rng()
    return g.uniform(low=-a, high=a, size=shape).astype(dtype)

def bias_zeros(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)
