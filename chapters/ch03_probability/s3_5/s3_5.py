"""
Section 3.5 — Conditional Probability (NumPy-only).

Re-exports helpers from this section so other code can import from one place.
"""
from .conditionals import (
    pmf_from_counts,
    marginal_from_joint,
    cond_pmf,                       # conditional table from joint
    joint_from_cond_and_prior,      # reconstruct joint from conditional + prior
    law_total_probability_check,    # sanity: sum_x P(x|y)P(y) == P(x)
)

__all__ = [
    "pmf_from_counts", "marginal_from_joint", "cond_pmf",
    "joint_from_cond_and_prior", "law_total_probability_check",
]
