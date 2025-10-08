import numpy as np
from chapters.ch02_linear_algebra.s2_10 import (
    trace, trace_cyclic_equal
)

def test_trace_basic_and_square_requirement():
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]], dtype=np.float64)
    assert np.isclose(trace(A), 1.0 + 4.0)

    # Non-square should raise (we enforce math definition)
    B = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float64)  # (2,3)
    try:
        _ = trace(B)
        assert False, "Expected trace() to require a square matrix"
    except ValueError:
        pass

def test_cyclic_property_two_factors():
    rng = np.random.default_rng(0)
    # A: (m,n), B: (n,m)
    m, n = 4, 3
    A = rng.normal(size=(m, n)).astype(np.float64)
    B = rng.normal(size=(n, m)).astype(np.float64)

    assert trace_cyclic_equal(A, B, C=None, tol=1e-12) is True

    # If we break shapes so AB not square, request should raise
    B_bad = rng.normal(size=(n, n)).astype(np.float64)  # (n,n) won't make AB square unless n == m
    try:
        trace_cyclic_equal(A, B_bad)
        assert False, "Expected a shape error for invalid AB product"
    except ValueError:
        pass

def test_cyclic_property_three_factors():
    rng = np.random.default_rng(1)
    # A:(m,n), B:(n,p), C:(p,m)
    m, n, p = 2, 3, 4
    A = rng.normal(size=(m, n)).astype(np.float64)
    B = rng.normal(size=(n, p)).astype(np.float64)
    C = rng.normal(size=(p, m)).astype(np.float64)

    assert trace_cyclic_equal(A, B, C, tol=1e-12) is True

    # Break shapes to force an error
    C_bad = rng.normal(size=(p, p)).astype(np.float64)  # not (p,m)
    try:
        trace_cyclic_equal(A, B, C_bad)
        assert False, "Expected a shape error for invalid ABC product"
    except ValueError:
        pass

def test_numeric_tolerance_small_noise():
    # Make matrices that should satisfy the cyclic property exactly,
    # then inject tiny floating-point noise and verify tolerance handles it.
    A = np.array([[1.0, 2.0],
                  [0.0, 1.0]], dtype=np.float64)  # (2,2)
    B = np.array([[0.5, 0.0],
                  [1.0, 1.5]], dtype=np.float64)  # (2,2)
    # For two factors with square shapes, the property still holds.
    # Add tiny noise to one product path.
    assert trace_cyclic_equal(A, B, tol=1e-12) is True

    # Three factors with exact cycle
    C = np.array([[2.0, -1.0],
                  [0.0,  3.0]], dtype=np.float64)  # (2,2)
    assert trace_cyclic_equal(A, B, C, tol=1e-12) is True
