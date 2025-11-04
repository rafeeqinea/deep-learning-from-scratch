import numpy as np
from chapters.ch06_feedforward.api import (
    cross_entropy_from_logits, ce_grad_wrt_logits
)
from chapters.ch04_numerical.api import central_diff_grad

def test_ce_grad_identity_matches_numerical_with_ids():
    rng = np.random.default_rng(123)
    B, C = 3, 4
    logits = rng.normal(size=(B, C)).astype(np.float64)
    targets = rng.integers(0, C, size=(B,))

    analytic = ce_grad_wrt_logits(logits, targets)

    def f(Lflat):
        L = Lflat.reshape(B, C)
        return cross_entropy_from_logits(L, targets)

    num = central_diff_grad(f, logits.copy(), eps=1e-6)
    assert np.allclose(analytic, num, rtol=5e-5, atol=5e-7)

def test_ce_grad_identity_matches_numerical_with_onehot():
    rng = np.random.default_rng(7)
    B, C = 2, 3
    logits = rng.normal(size=(B, C)).astype(np.float64)
    idx = rng.integers(0, C, size=(B,))
    onehot = np.eye(C)[idx]

    analytic = ce_grad_wrt_logits(logits, onehot)

    def f(Lflat):
        L = Lflat.reshape(B, C)
        return cross_entropy_from_logits(L, onehot)

    num = central_diff_grad(f, logits.copy(), eps=1e-6)
    assert np.allclose(analytic, num, rtol=5e-5, atol=5e-7)
