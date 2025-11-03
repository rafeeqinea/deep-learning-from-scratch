import numpy as np
from chapters.ch04_numerical.s4_1.floating_point import (
    machine_eps, ulp, exp_safe_bounds, kahan_sum, tiny_perturbation_is_ignored
)

def test_machine_eps_matches_numpy():
    assert machine_eps(np.float64) == np.finfo(np.float64).eps

def test_ulp_matches_spacing():
    x = np.array([0.0, 1.0, 10.0, 1e6], dtype=np.float64)
    assert np.allclose(ulp(x), np.spacing(x))

def test_exp_safe_bounds_behavior():
    min_x, max_x = exp_safe_bounds(np.float64)
    a = np.exp(np.array([min_x], dtype=np.float64))[0]
    b = np.exp(np.array([max_x], dtype=np.float64))[0]
    # inside thresholds should be finite
    assert np.isfinite(a) and np.isfinite(b)
    # outside thresholds under/over-flows as expected
    under = np.exp(np.array([min_x - 2.0], dtype=np.float64))[0]
    over  = np.exp(np.array([max_x + 2.0], dtype=np.float64))[0]
    assert under == 0.0
    assert np.isinf(over) and over > 0

def test_kahan_sum_beats_naive_on_pathological_sequence():
    # classic cancellation: naive sum loses the small term
    x = np.array([1e16, 1.0, -1e16], dtype=np.float64)
    naive = float(np.sum(x))
    kahan = float(kahan_sum(x))
    # naive is 0.0 on many platforms, Kahan recovers ~1.0
    assert abs(kahan - 1.0) < 1e-9
    assert abs(naive - 1.0) > 1e-6

def test_tiny_perturbation_is_ignored_float32():
    assert tiny_perturbation_is_ignored(1.0, dtype=np.float32) is True
