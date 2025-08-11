# tests/linear_algebra/test_matrices.py
import pytest
from core.linear_algebra.matrices import create_matrix, shape, transpose, add_matrices

def test_create_matrix():
    m = create_matrix([[1, 2], [3, 4]])
    assert m == [[1, 2], [3, 4]]

def test_shape():
    m = [[1, 2, 3], [4, 5, 6]]
    assert shape(m) == (2, 3)

def test_transpose():
    m = [[1, 2], [3, 4], [5, 6]]
    t = transpose(m)
    assert t == [[1, 3, 5], [2, 4, 6]]

def test_add_matrices():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    result = add_matrices(a, b)
    assert result == [[6, 8], [10, 12]]
