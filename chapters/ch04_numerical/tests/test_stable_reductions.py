import numpy as np
from chapters.ch04_numerical.s4_4.stable_reductions import logsumexp, log_softmax, softplus

def test_logsumexp_matches_naive_on_small_numbers():
    x = np.array([[0.1, 0.2, -0.3],
                  [1.0, -2.0, 0.0]], dtype=np.float64)
    naive = np.log(np.sum(np.exp(x), axis=1))
    stable = logsumexp(x, axis=1)
    assert np.allclose(naive, stable, rtol=1e-12, atol=1e-12)

def test_logsumexp_is_stable_on_large_values():
    x = np.array([[1000.0, 1001.0, 999.5],
                  [800.0, 799.9, 800.1]], dtype=np.float64)
    stable = logsumexp(x, axis=1)
    assert np.isfinite(stable).all()

def test_log_softmax_shift_invariance_rows():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 7))
    shift = rng.normal(size=(5, 1))  # per-row shift
    lsm1 = log_softmax(logits, axis=1)
    lsm2 = log_softmax(logits + shift, axis=1)
    assert np.allclose(lsm1, lsm2, atol=1e-12)

def test_log_softmax_exp_normalizes_to_one():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4, 6))
    lsm = log_softmax(logits, axis=1)
    probs = np.exp(lsm)
    rowsums = probs.sum(axis=1)
    assert np.allclose(rowsums, np.ones_like(rowsums), atol=1e-12)

def test_softplus_matches_naive_on_moderate_values():
    x = np.linspace(-10, 10, 101, dtype=np.float64)
    naive = np.log(1.0 + np.exp(x))  # safe in this range
    sp = softplus(x)
    assert np.allclose(sp, naive, rtol=1e-12, atol=1e-12)

def test_softplus_extremes_behave():
    big = np.array([1000.0], dtype=np.float64)
    small = np.array([-1000.0], dtype=np.float64)
    assert np.allclose(softplus(big), big, rtol=1e-12, atol=1e-12)  # ~x
    assert np.allclose(softplus(small), 0.0, rtol=0, atol=1e-12)    # ~0
