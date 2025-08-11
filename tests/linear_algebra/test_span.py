# tests/linear_algebra/test_span.py
import pytest
from core.linear_algebra.span import rank, are_linearly_independent

def test_rank():
    """Tests the rank calculation."""
    # Full rank 2x3 matrix
    A = [[1, 2, 3], [4, 5, 6]]
    assert rank(A) == 2
    
    # Rank deficient 3x2 matrix
    B = [[1, 2], [2, 4], [3, 6]]
    assert rank(B) == 1
    
    # Square full rank matrix
    C = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
    assert rank(C) == 3
    
    # Square singular matrix
    D = [[1, 2, 3], [2, 4, 6], [4, 5, 6]]
    assert rank(D) == 2

def test_are_linearly_independent_true():
    """Tests a set of linearly independent vectors."""
    # Standard basis vectors in R^3
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    v3 = [0, 0, 1]
    assert are_linearly_independent([v1, v2, v3]) is True
    
    # Another independent set
    u1 = [1, 1]
    u2 = [1, -1]
    assert are_linearly_independent([u1, u2]) is True

def test_are_linearly_independent_false():
    """Tests a set of linearly dependent vectors."""
    v1 = [1, 2, 3]
    v2 = [2, 4, 6] # v2 is 2*v1
    assert are_linearly_independent([v1, v2]) is False
    
    # One vector is a combination of the others
    u1 = [1, 0, 0]
    u2 = [0, 1, 0]
    u3 = [2, 3, 0] # u3 is 2*u1 + 3*u2
    assert are_linearly_independent([u1, u2, u3]) is False

def test_more_vectors_than_dimensions():
    """Tests that more vectors than dimensions must be dependent."""
    v1 = [1, 0]
    v2 = [0, 1]
    v3 = [1, 1]
    assert are_linearly_independent([v1, v2, v3]) is False

def test_are_linearly_independent_dimension_mismatch():
    """Tests that vectors must have the same dimension."""
    v1 = [1, 2]
    v2 = [1, 2, 3]
    with pytest.raises(ValueError, match="same dimension"):
        are_linearly_independent([v1, v2])