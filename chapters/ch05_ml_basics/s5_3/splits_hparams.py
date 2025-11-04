import numpy as np
from typing import Tuple, Optional

def _split_indices(n: int,
                   ratios: Tuple[float, float, float],
                   rng: Optional[np.random.Generator]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError("ratios must sum to 1.0")
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    n_train = int(round(n * r_train))
    n_val   = int(round(n * r_val))
    n_test  = n - n_train - n_val
    i_train = idx[:n_train]
    i_val   = idx[n_train:n_train+n_val]
    i_test  = idx[n_train+n_val:]
    return i_train, i_val, i_test

def train_val_test_split(X: np.ndarray,
                         y: np.ndarray,
                         ratios: Tuple[float,float,float] = (0.8, 0.1, 0.1),
                         shuffle: bool = True,
                         seed: Optional[int] = None):
    """
    Split arrays into train/val/test according to ratios.
    Returns: (Xtr, ytr), (Xv, yv), (Xte, yte)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same first dimension")
    rng = np.random.default_rng(seed) if (shuffle or seed is not None) else None
    i_tr, i_v, i_te = _split_indices(X.shape[0], ratios, rng)
    return (X[i_tr], y[i_tr]), (X[i_v], y[i_v]), (X[i_te], y[i_te])
