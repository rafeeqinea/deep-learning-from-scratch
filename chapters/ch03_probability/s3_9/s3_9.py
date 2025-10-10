"""
Section 3.9 — Likelihood, ML, MAP (NumPy-only).

Re-exports helpers from this section.
"""
from .ml_map import (
    nll_categorical_from_logits,   # multiclass NLL (stable)
    nll_binary_from_logits,        # binary NLL (stable)
    l2_penalty,                    # 1/2 * lambda * ||W||_2^2
    nll_with_l2,                   # NLL + L2 helper
)

__all__ = [
    "nll_categorical_from_logits", "nll_binary_from_logits",
    "l2_penalty", "nll_with_l2",
]
