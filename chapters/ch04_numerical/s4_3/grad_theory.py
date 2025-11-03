import numpy as np
from typing import Callable, Tuple

def make_ls_quadratic(A: np.ndarray, b: np.ndarray):
    """
    Build least-squares quadratic:
        f(x) = 0.5 * ||A x - b||^2
        grad(x) = A^T (A x - b)
    Returns (f, grad).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        r = A @ x - b
        return 0.5 * float(np.dot(r, r))

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        r = A @ x - b
        return A.T @ r

    return f, grad

def lipschitz_const_from_A(A: np.ndarray) -> float:
    """
    For f(x) = 0.5 ||A x - b||^2, grad is L-Lipschitz with L = sigma_max(A)^2.
    """
    A = np.asarray(A, dtype=np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] ** 2)

def descent_lemma_gap(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    L: float,
) -> float:
    """
    Compute gap = f(y) - [ f(x) + grad(x)^T (y-x) + (L/2)||y-x||^2 ].
    For L-smooth f, gap <= 0. (Numerically may be ~1e-12).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = grad(x)
    rhs = f(x) + float(np.dot(g, (y - x))) + 0.5 * L * float(np.dot(y - x, y - x))
    return float(f(y) - rhs)
