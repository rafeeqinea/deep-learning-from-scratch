"""
Section 2.12 — Principal Components Analysis (NumPy-only).

This file re-exports the small helper functions from this section so that
other code can import them from one place.
"""
from .pca import (
    center,              # subtract column means
    pca_svd,             # thin SVD of centered data
    explained_variance,  # variance ratio per component and cumulative
    project_to_k,        # X_centered -> Z_k in k dims
    reconstruct_from_k,  # Z_k -> X_hat in original feature space
)

__all__ = [
    "center", "pca_svd", "explained_variance",
    "project_to_k", "reconstruct_from_k",
]
