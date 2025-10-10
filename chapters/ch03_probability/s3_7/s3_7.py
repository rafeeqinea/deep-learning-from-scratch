"""
Section 3.7 — Expectation, Variance, Covariance (NumPy-only).

Re-exports helpers from this section.
"""
from .expectation import (
    expectation_discrete, variance_discrete,
    covariance_from_joint2,       # Cov[X,Y] from 2D joint table
    mean_empirical, var_empirical,
    cov_empirical,                # sample covariance matrix (ddof=1)
)

__all__ = [
    "expectation_discrete", "variance_discrete", "covariance_from_joint2",
    "mean_empirical", "var_empirical", "cov_empirical",
]
