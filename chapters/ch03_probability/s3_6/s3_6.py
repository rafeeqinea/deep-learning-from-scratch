"""
Section 3.6 — Chain Rule of Probability (NumPy-only).

Re-exports helpers from this section.
"""
from .factorization import (
    marginalize_nd,
    chain_rule_product_equals_joint_3d,  # verify P(a,b,c) = P(a)P(b|a)P(c|a,b)
)

__all__ = [
    "marginalize_nd", "chain_rule_product_equals_joint_3d",
]
