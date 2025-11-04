import numpy as np
from typing import Tuple, Union

Array = np.ndarray

def as_one_hot(labels: Array, num_classes: int, dtype=np.float32) -> Array:
    """
    Convert integer class ids shape (B,) to one-hot shape (B, C).
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D of class ids")
    B = labels.shape[0]
    out = np.zeros((B, num_classes), dtype=dtype)
    out[np.arange(B), labels.astype(int)] = 1
    return out

def _targets_to_ids(targets: Array) -> Array:
    """
    Accepts integer ids (B,) or one-hot (B,C) and returns ids (B,).
    """
    t = np.asarray(targets)
    if t.ndim == 1:
        return t.astype(int)
    if t.ndim == 2:
        return np.argmax(t, axis=1).astype(int)
    raise ValueError("targets must be (B,) or (B,C)")

def accuracy_from_logits(logits: Array, targets: Array) -> float:
    """
    Top-1 accuracy. logits: (B,C), targets: (B,) ids or (B,C) one-hot.
    """
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError("logits must be 2D (B,C)")
    preds = np.argmax(logits, axis=1)
    ids = _targets_to_ids(targets)
    if ids.shape[0] != preds.shape[0]:
        raise ValueError("batch size mismatch between logits and targets")
    return float(np.mean(preds == ids))
