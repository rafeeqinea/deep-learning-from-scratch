import numpy as np
from chapters.ch03_probability.s3_10 import (
    logsumexp, softmax, log_softmax, one_hot,
    cross_entropy_from_logits, binary_cross_entropy_from_logits,
    log_sigmoid, sigmoid
)

def test_logsumexp_vs_naive_and_stability():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100,))
    lse = logsumexp(x)
    naive = np.log(np.sum(np.exp(x)))
    assert np.isclose(lse, naive)

    big = np.array([1000.0, 1001.0, 999.0])
    lse_big = logsumexp(big)
    assert np.isfinite(lse_big)

def test_softmax_properties_and_shift_invariance():
    rng = np.random.default_rng(1)
    L = rng.normal(size=(5, 10))
    P = softmax(L, axis=1)
    assert np.allclose(P.sum(axis=1), 1.0)

    # shift invariance: softmax(L) == softmax(L + c)
    c = 123.456
    P2 = softmax(L + c, axis=1)
    assert np.allclose(P, P2, atol=1e-12)

def test_log_softmax_and_cross_entropy_index_targets():
    rng = np.random.default_rng(2)
    L = rng.normal(size=(6, 7))
    y = rng.integers(low=0, high=7, size=(6,))
    # CE equals -log p(correct)
    CE = cross_entropy_from_logits(L, y, reduction="none")
    LS = log_softmax(L, axis=1)
    ref = -LS[np.arange(6), y]
    assert np.allclose(CE, ref)

def test_cross_entropy_one_hot_targets():
    rng = np.random.default_rng(3)
    L = rng.normal(size=(4, 5))
    y_idx = np.array([0, 2, 1, 3])
    Y = one_hot(y_idx, 5, dtype=np.float64)
    ce_oh = cross_entropy_from_logits(L, Y, reduction="none")
    ce_idx = cross_entropy_from_logits(L, y_idx, reduction="none")
    assert np.allclose(ce_oh, ce_idx)

def test_binary_cross_entropy_from_logits_matches_logistic_loss():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(8,))
    y = rng.integers(low=0, high=2, size=(8,)).astype(float)
    bce = binary_cross_entropy_from_logits(z, y, reduction="none")

    # logistic loss: softplus(z) - y*z
    sp = np.where(z > 20, z, np.log1p(np.exp(z)))
    ref = sp - y * z
    assert np.allclose(bce, ref)

def test_sigmoid_and_log_sigmoid_sanity():
    z = np.array([-100.0, 0.0, 100.0])
    s = sigmoid(z)
    ls = log_sigmoid(z)
    assert np.allclose(s, [0.0, 0.5, 1.0], atol=1e-12)
    assert np.allclose(np.exp(ls), s, atol=1e-12)
