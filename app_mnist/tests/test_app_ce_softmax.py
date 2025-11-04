import numpy as np
from chapters.ch06_feedforward.api import cross_entropy_from_logits, softmax_from_logits
from chapters.ch04_numerical.api import log_softmax

def test_ce_shift_invariance_and_softmax_rowsum():
    rng = np.random.default_rng(0)
    B, C = 8, 5
    logits = rng.normal(size=(B, C))
    y = rng.integers(0, C, size=(B,))
    ce0 = cross_entropy_from_logits(logits, y)
    shift = rng.normal(size=(B,1))
    ce1 = cross_entropy_from_logits(logits + shift, y)
    assert np.allclose(ce0, ce1, atol=1e-12)

    p = softmax_from_logits(logits, axis=1)
    assert np.allclose(p.sum(axis=1), np.ones(B))
    # matches exp(log_softmax)
    assert np.allclose(p, np.exp(log_softmax(logits, axis=1)))
