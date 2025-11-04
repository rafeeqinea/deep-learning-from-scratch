import numpy as np
from typing import Tuple
from chapters.ch04_numerical.api import log_softmax

Array = np.ndarray

def _targets_to_ids_or_onehot(targets: Array, num_classes: int):
    """
    Accept labels as (B,) int ids or (B,C) one-hot and return (ids, onehot).
    """
    t = np.asarray(targets)
    if t.ndim == 1:
        ids = t.astype(int)
        oh = np.zeros((t.shape[0], num_classes), dtype=np.float64)
        oh[np.arange(t.shape[0]), ids] = 1.0
        return ids, oh
    if t.ndim == 2:
        ids = np.argmax(t, axis=1).astype(int)
        return ids, t.astype(np.float64)
    raise ValueError("targets must be (B,) or (B,C)")

def cross_entropy_from_logits(logits: Array, targets: Array) -> float:
    """
    Mean cross-entropy using log-softmax for numerical stability.
    logits: (B,C) float
    targets: (B,) int or (B,C) one-hot
    Returns scalar CE (mean over batch).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must be 2D (B,C)")
    B, C = logits.shape
    _, onehot = _targets_to_ids_or_onehot(targets, num_classes=C)
    lsm = log_softmax(logits, axis=1)           # (B,C)
    ce = -np.sum(onehot * lsm) / B
    return float(ce)

def softmax_from_logits(logits: Array, axis: int = -1) -> Array:
    """
    Softmax via exp(log_softmax) for stability.
    """
    lsm = log_softmax(np.asarray(logits, dtype=np.float64), axis=axis)
    return np.exp(lsm)
