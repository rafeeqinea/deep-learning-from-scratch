# Expectation, variance, and covariance (NumPy only).
import numpy as np

def expectation_discrete(values, pmf):
    """
    E[X] for a 1-D discrete variable.
      values: shape (n,)
      pmf:    shape (n,), sums to 1
    """
    x = np.asarray(values, dtype=float)
    p = np.asarray(pmf, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.shape[0] != p.shape[0]:
        raise ValueError("values and pmf must be 1-D with same length.")
    s = p.sum()
    if not np.isfinite(s) or abs(s - 1.0) > 1e-8:
        raise ValueError("pmf must sum to 1.")
    return float((x * p).sum())

def variance_discrete(values, pmf):
    """
    Var[X] = E[(X - E[X])^2] for 1-D discrete.
    """
    mu = expectation_discrete(values, pmf)
    x = np.asarray(values, dtype=float)
    p = np.asarray(pmf, dtype=float)
    return float(((x - mu) ** 2 * p).sum())

def covariance_from_joint2(values_x, values_y, joint_xy):
    """
    Cov[X,Y] from a 2-D joint table over values_x (len n) and values_y (len m).
    joint_xy has shape (n, m) and sums to 1.
    """
    vx = np.asarray(values_x, dtype=float)
    vy = np.asarray(values_y, dtype=float)
    P = np.asarray(joint_xy, dtype=float)
    if P.shape != (vx.size, vy.size):
        raise ValueError("shape mismatch.")
    s = P.sum()
    if not np.isfinite(s) or abs(s - 1.0) > 1e-8:
        raise ValueError("joint must sum to 1.")

    Ex = (vx * P.sum(axis=1)).sum()
    Ey = (vy * P.sum(axis=0)).sum()
    # E[XY]
    XY = (vx.reshape(-1,1) * vy.reshape(1,-1))
    EXY = (XY * P).sum()
    return float(EXY - Ex * Ey)

def mean_empirical(X, axis=0):
    X = np.asarray(X, dtype=float)
    return X.mean(axis=axis)

def var_empirical(X, axis=0, ddof=1):
    """
    Unbiased sample variance by default (ddof=1).
    """
    X = np.asarray(X, dtype=float)
    return X.var(axis=axis, ddof=ddof)

def cov_empirical(X):
    """
    Sample covariance matrix (ddof=1) for rows=samples, cols=features.
    Returns shape (n_features, n_features).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (samples, features).")
    # np.cov expects variables in rows by default -> rowvar=False
    return np.cov(X, rowvar=False, ddof=1)
