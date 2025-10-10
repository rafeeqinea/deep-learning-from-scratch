import numpy as np
from chapters.ch03_probability.s3_9 import (
    nll_categorical_from_logits,
    nll_binary_from_logits,
    l2_penalty,
    nll_with_l2,
)
from chapters.ch03_probability.s3_10 import (
    one_hot,
    cross_entropy_from_logits,
    binary_cross_entropy_from_logits,
)

def test_nll_categorical_matches_cross_entropy_indices():
    rng = np.random.default_rng(0)
    B, C = 8, 5
    logits = rng.normal(size=(B, C))
    y_idx = rng.integers(low=0, high=C, size=B)

    # Reference via CE from logits
    ref_none = cross_entropy_from_logits(logits, y_idx, reduction="none")
    ref_mean = cross_entropy_from_logits(logits, y_idx, reduction="mean")
    ref_sum  = cross_entropy_from_logits(logits, y_idx, reduction="sum")

    # NLL should match CE
    nll_none = nll_categorical_from_logits(logits, y_idx, reduction="none")
    nll_mean = nll_categorical_from_logits(logits, y_idx, reduction="mean")
    nll_sum  = nll_categorical_from_logits(logits, y_idx, reduction="sum")

    assert np.allclose(nll_none, ref_none)
    assert np.isclose(nll_mean, ref_mean)
    assert np.isclose(nll_sum,  ref_sum)
    assert isinstance(nll_mean, float) and isinstance(nll_sum, float)

def test_nll_categorical_one_hot_equals_indices_and_reductions():
    rng = np.random.default_rng(1)
    B, C = 6, 7
    L = rng.normal(size=(B, C))
    y_idx = rng.integers(low=0, high=C, size=B)
    Y = one_hot(y_idx, C, dtype=np.float64)

    nll_idx = nll_categorical_from_logits(L, y_idx, reduction="none")
    nll_oh  = nll_categorical_from_logits(L, Y, reduction="none")
    assert np.allclose(nll_idx, nll_oh)

    # reduction modes
    assert isinstance(nll_categorical_from_logits(L, y_idx, reduction="mean"), float)
    assert isinstance(nll_categorical_from_logits(L, y_idx, reduction="sum"), float)

def test_nll_binary_matches_bce_and_broadcast():
    rng = np.random.default_rng(2)
    B = 10
    z_vec = rng.normal(size=(B,))
    y = rng.integers(low=0, high=2, size=B).astype(float)

    # Reference logistic loss
    ref = binary_cross_entropy_from_logits(z_vec, y, reduction="none")
    nll = nll_binary_from_logits(z_vec, y, reduction="none")
    assert np.allclose(nll, ref)

    # Broadcasting: logits (B,1) with y (B,) should work
    z_col = z_vec.reshape(B, 1)
    nll_bc = nll_binary_from_logits(z_col, y, reduction="none")
    assert np.allclose(nll_bc.reshape(-1), ref.reshape(-1))

    # reductions return floats
    assert isinstance(nll_binary_from_logits(z_vec, y, reduction="mean"), float)
    assert isinstance(nll_binary_from_logits(z_vec, y, reduction="sum"),  float)

def test_l2_penalty_and_nll_with_l2():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(4, 3))
    lam = 0.1
    l2 = l2_penalty(W, lam)
    # 0.5 * lam * ||W||^2
    ref = 0.5 * lam * float(np.sum(W * W))
    assert np.isclose(l2, ref)

    # Add L2 to a base NLL
    base = 2.345
    total = nll_with_l2(base, W, lam)
    assert np.isclose(total, base + ref)

def test_nll_numerical_stability_large_logits():
    # Extremely large/small logits should not overflow (rely on stable log-softmax/softplus)
    big_pos = np.array([[1000.0, 0.0, -1000.0]])
    big_neg = np.array([[-1000.0, 0.0, 1000.0]])
    y_idx = np.array([0])

    # Categorical NLL: finite and consistent
    nll1 = nll_categorical_from_logits(big_pos, y_idx, reduction="mean")
    nll2 = nll_categorical_from_logits(big_neg, np.array([2]), reduction="mean")
    assert np.isfinite(nll1) and np.isfinite(nll2)

    # Binary NLL with huge logits
    z = np.array([1000.0, -1000.0])
    y = np.array([1.0, 0.0])
    bce = nll_binary_from_logits(z, y, reduction="none")
    assert np.all(np.isfinite(bce))
