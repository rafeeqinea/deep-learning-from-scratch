import numpy as np
import random
from typing import Iterator, Optional, Tuple

def set_all_seeds(seed: int) -> None:
    """
    Deterministic seeding for numpy + random (Python).
    """
    random.seed(seed)
    np.random.seed(seed)  # also ok: np.random.default_rng(seed) per-call

def num_batches(n_samples: int, batch_size: int, drop_last: bool = False) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    q, r = divmod(n_samples, batch_size)
    return q if (drop_last or r == 0) else q + 1

def batch_iterator(X: np.ndarray,
                   y: Optional[np.ndarray],
                   batch_size: int,
                   shuffle: bool = True,
                   drop_last: bool = False,
                   seed: Optional[int] = None) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Yield mini-batches (Xb, yb) of given batch_size.
    If y is None, yield (Xb, None).
    """
    X = np.asarray(X)
    n = X.shape[0]
    if y is not None and y.shape[0] != n:
        raise ValueError("X and y must have same length")
    rng = np.random.default_rng(seed) if (shuffle or seed is not None) else None
    indices = np.arange(n)
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = start + batch_size
        if end > n and drop_last:
            break
        batch_idx = indices[start:end]
        Xb = X[batch_idx]
        yb = None if y is None else y[batch_idx]
        yield Xb, yb
