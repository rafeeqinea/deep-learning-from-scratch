import numpy as np
from typing import Callable, Tuple

def central_diff_grad(f: Callable[[np.ndarray], float],
                      x: np.ndarray,
                      eps: float = 1e-5) -> np.ndarray:
    """
    Numerical gradient of scalar function f at x via central differences.
    x: any shape, float64 recommended.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad

def check_grad(f: Callable[[np.ndarray], float],
               x: np.ndarray,
               analytical_grad: np.ndarray,
               rtol: float = 1e-4,
               atol: float = 1e-6,
               eps: float = 1e-5) -> Tuple[bool, float]:
    """
    Compare analytical_grad to numerical grad.
    Returns: (ok, max_abs_diff)
    """
    num = central_diff_grad(f, x.astype(np.float64), eps=eps)
    diff = np.abs(num - analytical_grad.astype(np.float64))
    return (np.allclose(num, analytical_grad, rtol=rtol, atol=atol),
            float(np.max(diff)))
