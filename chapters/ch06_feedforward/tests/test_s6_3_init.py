import numpy as np
from chapters.ch06_feedforward.api import glorot_uniform, bias_zeros

def test_glorot_uniform_bounds_and_stats():
    rng = np.random.default_rng(42)
    shape = (200, 150)
    W = glorot_uniform(shape, rng=rng, dtype=np.float64)
    fan_in, fan_out = shape
    a = np.sqrt(6.0 / (fan_in + fan_out))
    assert W.shape == shape
    # values within [-a, a]
    assert np.max(W) <= a + 1e-12 and np.min(W) >= -a - 1e-12
    # mean ~ 0 and var ~ a^2/3 (allow generous tolerance)
    m = float(W.mean())
    v = float(W.var())
    assert abs(m) < 1e-2
    assert abs(v - (a*a/3.0)) / (a*a/3.0) < 0.25  # 25% tolerance

def test_bias_zeros():
    b = bias_zeros((10,), dtype=np.float32)
    assert b.shape == (10,)
    assert b.dtype == np.float32
    assert float(b.sum()) == 0.0
