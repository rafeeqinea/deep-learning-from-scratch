import numpy as np
from chapters.ch03_probability.s3_7 import (
    expectation_discrete, variance_discrete, covariance_from_joint2,
    mean_empirical, var_empirical, cov_empirical
)

def test_discrete_expectation_and_variance():
    # Fair die on {1..6}
    x = np.arange(1, 7, dtype=float)
    p = np.ones_like(x) / x.size
    mu = expectation_discrete(x, p)
    var = variance_discrete(x, p)
    assert np.isclose(mu, 3.5)
    assert np.isclose(var, 35/12)

def test_covariance_from_joint2():
    # Simple correlated 2x2 example
    vx = np.array([0., 1.])
    vy = np.array([0., 1.])
    P = np.array([[0.4, 0.1],
                  [0.1, 0.4]])
    P = P / P.sum()
    cov = covariance_from_joint2(vx, vy, P)  # should be positive
    assert cov > 0

def test_empirical_stats_match_numpy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    assert np.allclose(mean_empirical(X), X.mean(axis=0))
    assert np.allclose(var_empirical(X), X.var(axis=0, ddof=1))
    assert np.allclose(cov_empirical(X), np.cov(X, rowvar=False, ddof=1))
