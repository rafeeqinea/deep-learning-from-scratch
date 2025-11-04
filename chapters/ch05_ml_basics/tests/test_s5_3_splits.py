import numpy as np
import pytest
from chapters.ch05_ml_basics.api import train_val_test_split

def test_split_shapes_and_disjoint():
    n = 13
    X = np.arange(n*2).reshape(n,2)
    y = np.arange(n)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(
        X, y, ratios=(0.6, 0.2, 0.2), shuffle=True, seed=123
    )
    assert Xtr.shape[0] + Xv.shape[0] + Xte.shape[0] == n
    assert set(ytr).isdisjoint(yv) and set(ytr).isdisjoint(yte) and set(yv).isdisjoint(yte)
    # determinism with same seed
    (_, ytr2), (_, yv2), (_, yte2) = train_val_test_split(
        X, y, ratios=(0.6, 0.2, 0.2), shuffle=True, seed=123
    )
    assert np.array_equal(ytr, ytr2) and np.array_equal(yv, yv2) and np.array_equal(yte, yte2)

def test_split_raises_on_length_mismatch():
    X = np.zeros((5,3))
    y = np.zeros((4,))
    with pytest.raises(ValueError):
        train_val_test_split(X, y)

def test_split_ratios_must_sum_to_one():
    X = np.zeros((10,2))
    y = np.zeros((10,))
    with pytest.raises(ValueError):
        train_val_test_split(X, y, ratios=(0.5, 0.5, 0.1))
