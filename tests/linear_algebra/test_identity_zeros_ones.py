# tests/linear_algebra/test_identity_zeros_ones.py
import pytest
from core.linear_algebra.identity_zeros_ones import identity, zeros, ones

def test_identity():
    """Tests creation of a 3x3 identity matrix."""
    expected = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    assert identity(3) == expected

def test_identity_one_by_one():
    """Tests the edge case of a 1x1 identity matrix."""
    assert identity(1) == [[1]]

def test_identity_raises_for_non_positive_size():
    """Tests that identity matrix size must be positive."""
    with pytest.raises(ValueError):
        identity(0)
    with pytest.raises(ValueError):
        identity(-5)

def test_zeros():
    """Tests creation of a 2x3 matrix of zeros."""
    expected = [
        [0, 0, 0],
        [0, 0, 0]
    ]
    assert zeros(2, 3) == expected

def test_ones():
    """Tests creation of a 3x2 matrix of ones."""
    expected = [
        [1, 1],
        [1, 1],
        [1, 1]
    ]
    assert ones(3, 2) == expected

def test_creation_functions_raise_for_non_positive_dims():
    """Tests that matrix dimensions must be positive."""
    with pytest.raises(ValueError):
        zeros(0, 2)
    with pytest.raises(ValueError):
        ones(2, -1)