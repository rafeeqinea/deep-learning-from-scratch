import numpy as np

def cond2_via_svd(A: np.ndarray) -> float:
    """
    2-norm condition number cond_2(A) = sigma_max / sigma_min.
    Treats near-singular as singular using a relative threshold.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2D.")
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = float(s[0])
    sigma_min = float(s[-1])
    # Relative test: if smallest singular value is within eps of zero
    # compared to sigma_max, treat as singular.
    eps = np.finfo(np.float64).eps
    if sigma_min <= eps * sigma_max:
        return float("inf")
    return sigma_max / sigma_min
