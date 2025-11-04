import numpy as np
from chapters.ch06_feedforward.api import (
    cross_entropy_from_logits, softmax_from_logits
)
from chapters.ch04_numerical.api import log_softmax

def test_softmax_rows_sum_to_1_and_stable():
    logits = np.array([[1000.0, 1001.0, 999.0],
                       [  -5.0,   -4.0,  -6.0]], dtype=np.float64)
    p = softmax_from_logits(logits, axis=1)
    rowsum = p.sum(axis=1)
    assert np.allclose(rowsum, np.ones_like(rowsum), atol=1e-12)
    assert np.isfinite(p).all()

def test_ce_matches_logsoftmax_formula_and_shift_invariance():
    rng = np.random.default_rng(0)
    B, C = 5, 4
    logits = rng.normal(size=(B, C))
    targets = rng.integers(0, C, size=(B,))

    # CE via our API
    ce_api = cross_entropy_from_logits(logits, targets)

    # CE via stable formula: -mean(log_softmax[row, target])
    lsm = log_softmax(logits, axis=1)
    ce_ref = -float(np.mean(lsm[np.arange(B), targets]))
    assert np.allclose(ce_api, ce_ref, rtol=1e-12, atol=1e-12)

    # Shift invariance: add per-row constant -> CE unchanged
    shift = rng.normal(size=(B, 1))
    ce_shift = cross_entropy_from_logits(logits + shift, targets)
    assert np.allclose(ce_api, ce_shift, atol=1e-12)

def test_ce_accepts_one_hot_targets():
    logits = np.array([[0.1, 0.9, -0.2],
                       [2.0, -1.0, 0.0]], dtype=np.float64)
    ids = np.array([1, 0], dtype=int)
    onehot = np.eye(3)[ids]
    ce_ids = cross_entropy_from_logits(logits, ids)
    ce_oh  = cross_entropy_from_logits(logits, onehot)
    assert np.allclose(ce_ids, ce_oh, atol=1e-12)
