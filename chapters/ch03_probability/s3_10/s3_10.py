"""
Section 3.10 — Numerics for Probabilities (NumPy-only).

Re-exports helpers from this section (stable log-sum-exp, softmax, CE/BCE).
"""
from .prob_numerics import (
    logsumexp,
    softmax, log_softmax,
    one_hot,
    cross_entropy_from_logits,
    binary_cross_entropy_from_logits,
    log_sigmoid, sigmoid,
)

__all__ = [
    "logsumexp", "softmax", "log_softmax", "one_hot",
    "cross_entropy_from_logits", "binary_cross_entropy_from_logits",
    "log_sigmoid", "sigmoid",
]
