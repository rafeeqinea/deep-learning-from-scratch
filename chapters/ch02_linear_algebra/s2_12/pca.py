# PCA via SVD (NumPy only), written simply with clear comments.
import numpy as np
from chapters.ch02_linear_algebra.s2_1 import ensure_matrix


def center(X):
    """
    Center data matrix X by subtracting the column-wise mean.

    Shape convention:
      X: (m, n) = m samples (rows) × n features (cols)

    Returns:
      X_centered: (m, n)
      mu:         (n,)  the mean of each column (feature)
    """
    ensure_matrix(X)
    mu = X.mean(axis=0)
    Xc = X - mu
    return Xc, mu


def pca_svd(X_centered):
    """
    Compute the thin SVD of a *centered* data matrix:
        X_centered = U @ diag(S) @ VT

    Shapes (with full_matrices=False):
        X_centered: (m, n)
        U: (m, r)
        S: (r,)
        VT: (r, n)
      where r = min(m, n)

    Notes:
      - Principal axes (directions in feature space) are the rows of VT.
      - Scores (coordinates of samples in PC space) are Z = U * S.

    Important:
      This function expects X to be *centered*. Call center(X) first.
    """
    ensure_matrix(X_centered)
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    # Keep dtype of input for consistency
    U = U.astype(X_centered.dtype, copy=False)
    S = S.astype(X_centered.dtype, copy=False)
    VT = VT.astype(X_centered.dtype, copy=False)
    return U, S, VT


def explained_variance(S):
    """
    Return the per-component *variance ratio* and its cumulative.

    For centered data X with singular values S, the eigenvalues of the
    covariance matrix are proportional to S^2 (exactly (S^2)/(m-1)).
    The ratios therefore do not depend on dividing by (m-1).

    Inputs:
      S: (r,) singular values from SVD(X_centered)

    Returns:
      var_ratio: (r,) each entry in [0,1], sums to 1
      cum_ratio: (r,) cumulative sum of var_ratio (nondecreasing)
    """
    S = np.asarray(S)
    lam = S ** 2
    total = lam.sum()
    if total == 0:
        # All-zero data -> return zeros (avoid divide-by-zero)
        var_ratio = np.zeros_like(lam)
    else:
        var_ratio = lam / total
    cum_ratio = np.cumsum(var_ratio)
    return var_ratio, cum_ratio


def project_to_k(X_centered, VT, k):
    """
    Project centered data onto the top-k principal components.

    Inputs:
      X_centered: (m, n)
      VT:         (r, n)  rows are principal axes from SVD
      k:          int     number of components, 0 <= k <= r

    Returns:
      Z_k: (m, k)  sample coordinates in the top-k PC subspace

    Formula:
      Z_k = X_centered @ (VT[:k].T)
    """
    ensure_matrix(X_centered)
    ensure_matrix(VT)
    m, n = X_centered.shape
    r, n2 = VT.shape
    if n != n2:
        raise ValueError("project_to_k: X and VT feature dims differ: %r vs %r" % (X_centered.shape, VT.shape))
    k = int(k)
    if k < 0 or k > r:
        raise ValueError("project_to_k: k must be in [0, %d]" % r)
    if k == 0:
        return np.zeros((m, 0), dtype=X_centered.dtype)
    return X_centered @ VT[:k].T


def reconstruct_from_k(Z_k, VT, k):
    """
    Reconstruct data from top-k PCA coordinates back in the original space.

    Inputs:
      Z_k: (m, k)   top-k PC coordinates of rows
      VT:  (r, n)   principal axes (rows), from SVD
      k:   int

    Returns:
      X_hat_centered: (m, n)  reconstruction in the original feature space,
                               but still centered (add mu externally if needed)

    Formula:
      X_hat_centered = Z_k @ VT[:k]
    """
    ensure_matrix(Z_k)
    ensure_matrix(VT)
    m, k1 = Z_k.shape
    r, n = VT.shape
    k = int(k)
    if k != k1:
        raise ValueError("reconstruct_from_k: Z_k has k=%d, but argument k=%d" % (k1, k))
    if k < 0 or k > r:
        raise ValueError("reconstruct_from_k: k must be in [0, %d]" % r)
    if k == 0:
        return np.zeros((m, n), dtype=Z_k.dtype)
    return Z_k @ VT[:k]
