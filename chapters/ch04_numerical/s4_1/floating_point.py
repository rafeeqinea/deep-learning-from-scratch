import numpy as np
from typing import Tuple, Union, Optional

ArrayLike = Union[np.ndarray, float]

def machine_eps(dtype=np.float64) -> float:
    """
    Machine epsilon for the given dtype.
    """
    return float(np.finfo(dtype).eps)

def ulp(x: ArrayLike) -> np.ndarray:
    """
    Unit-in-the-last-place for each element of x.
    """
    x = np.asarray(x)
    return np.spacing(x)

def exp_safe_bounds(dtype=np.float64) -> Tuple[float, float]:
    """
    Finite range for exp(x) in the given dtype:
      min_x: at and below this, exp(x) underflows to EXACT zero
      max_x: above this, exp(x) overflows to +inf
    """
    finfo = np.finfo(dtype)
    # overflow threshold: log(max normal)
    max_x = float(np.log(finfo.max))
    # underflow-to-zero threshold: log(smallest positive subnormal)
    zero = np.array(0, dtype=dtype)
    one  = np.array(1, dtype=dtype)
    smallest_subnormal = np.nextafter(zero, one)  # > 0 but as small as representable
    min_x = float(np.log(smallest_subnormal))
    return min_x, max_x

def kahan_sum(x: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
    """
    Neumaier compensated summation (more robust than plain Kahan for mixed magnitudes).
    Returns float64 for accuracy.
    """
    x = np.asarray(x, dtype=np.float64)

    def neumaier(vec: np.ndarray) -> np.ndarray:
        s = 0.0
        c = 0.0
        for v in vec:
            t = s + v
            if abs(s) >= abs(v):
                c += (s - t) + v
            else:
                c += (v - t) + s
            s = t
        return np.array(s + c, dtype=np.float64)

    if axis is None:
        return neumaier(x.ravel())
    return np.apply_along_axis(lambda a: neumaier(a), axis, x)

def tiny_perturbation_is_ignored(a: float, dtype=np.float32) -> bool:
    """
    Returns True if adding eps/2 to 'a' in the given dtype leaves it unchanged.
    """
    eps = np.finfo(dtype).eps
    av = np.array(a, dtype=dtype)
    return bool((av + (eps / 2)).astype(dtype) == av)
