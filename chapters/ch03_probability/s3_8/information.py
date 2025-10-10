# Entropy, cross-entropy, and KL for discrete distributions (NumPy only).
import numpy as np

def _normalize_p(p):
    p = np.asarray(p, dtype=float)
    if np.any(p < -1e-12):
        raise ValueError("probabilities must be nonnegative.")
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("sum of probabilities must be positive/finite.")
    p = p / s
    return p

def _safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))

def entropy(p):
    """
    Shannon entropy H(P) = - sum p log p   (nats; base e)
    Zero terms are treated as 0 (via clipping).
    """
    p = _normalize_p(p)
    return float(-(p * _safe_log(p)).sum())

def cross_entropy(p, q):
    """
    Cross-entropy H(P, Q) = - sum p log q
    """
    p = _normalize_p(p)
    q = _normalize_p(q)
    return float(-(p * _safe_log(q)).sum())

def kl_divergence(p, q):
    """
    KL(P || Q) = sum p log (p / q)  >= 0
    """
    p = _normalize_p(p)
    q = _normalize_p(q)
    return float((p * (_safe_log(p) - _safe_log(q))).sum())
