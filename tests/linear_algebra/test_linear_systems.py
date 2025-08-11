# tests/linear_algebra/test_linear_systems.py
import pytest
from core.linear_algebra.linear_systems import solve

def test_solve_simple_2x2():
    """Tests a simple 2x2 system:
    2x + 3y = 8
     x + 2y = 5
    Solution: x=1, y=2
    """
    A = [[2, 3], [1, 2]]
    b = [8, 5]
    x = solve(A, b)
    assert len(x) == 2
    assert pytest.approx(x[0]) == 1
    assert pytest.approx(x[1]) == 2

def test_solve_3x3():
    """Tests a 3x3 system."""
    A = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
    b = [6, -4, 27]
    # Solution from online calculator: x=5, y=3, z=-2
    x = solve(A, b)
    assert len(x) == 3
    assert pytest.approx(x[0]) == 5
    assert pytest.approx(x[1]) == 3
    assert pytest.approx(x[2]) == -2

def test_solve_requires_pivoting():
    """Tests a system where the initial pivot is zero.
    0x + 1y = 2  ->  y = 2
    2x + 3y = 8  -> 2x + 3(2) = 8 -> 2x = 2 -> x = 1
    """
    A = [[0, 1], [2, 3]]
    b = [2, 8]
    x = solve(A, b)
    assert pytest.approx(x[0]) == 1
    assert pytest.approx(x[1]) == 2

def test_solve_singular_matrix():
    """Tests a system with a singular matrix (no unique solution)."""
    A = [[1, 1], [1, 1]] # Linearly dependent rows
    b = [2, 3]
    with pytest.raises(ValueError, match="Matrix is singular"):
        solve(A, b)

def test_solve_shape_mismatch_non_square():
    """Tests that A must be square."""
    A = [[1, 2, 3], [4, 5, 6]]
    b = [1, 2]
    with pytest.raises(ValueError, match="Matrix A must be square"):
        solve(A, b)

def test_solve_shape_mismatch_b_vector():
    """Tests that b must have the same dimension as A."""
    A = [[1, 2], [3, 4]]
    b = [1, 2, 3]
    with pytest.raises(ValueError, match="Shape mismatch"):
        solve(A, b)