"""
Section 3.8 — Information Theory (NumPy-only).

Re-exports helpers from this section.
"""
from .information import (
    entropy, cross_entropy, kl_divergence,
)

__all__ = ["entropy", "cross_entropy", "kl_divergence"]
