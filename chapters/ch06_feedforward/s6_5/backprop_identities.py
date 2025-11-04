import numpy as np
from typing import Tuple
from chapters.ch06_feedforward.s6_4.losses import softmax_from_logits

Array = np.ndarray

def _as_one_hot(labels: Array, num_classes: int, dtype=np.float64) -> Array:
    labels = np.asarray(labels).astype(int)
    B = labels.shape[0]
    out = np.zeros((B, num_classes), dtype=dtype)
    out[np.arange(B), labels] = 1.0
    return out

def _targets_to_onehot(targets: Array, num_classes: int) -> Array:
    t = np.asarray(targets)
    if t.ndim == 1:
        return _as_one_hot(t, num_classes)
    if t.ndim == 2:
        return t.astype(np.float64)
    raise ValueError("targets must be (B,) or (B,C)")

def ce_grad_wrt_logits(logits: Array, targets: Array) -> Array:
    """
    ∂(mean CE)/∂logits = (softmax(logits) - one_hot(targets)) / B
    Uses stable softmax computed from log_softmax.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must be 2D (B,C)")
    B, C = logits.shape
    probs = softmax_from_logits(logits, axis=1)  # (B,C)
    onehot = _targets_to_onehot(targets, num_classes=C)  # (B,C)
    grad = (probs - onehot) / B
    return grad
