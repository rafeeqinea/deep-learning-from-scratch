import pytest

from core.linear_algebra.matrices import create_matrix
from core.linear_algebra.matrix_vector_product import matrix_vector_product


def test_matrix_vector_product_happy():
    A = create_matrix([[1, 2, 3], [0, -1, 2]])
    v = [4, 5, 6]
    y = matrix_vector_product(A, v)
    assert y == [1 * 4 + 2 * 5 + 3 * 6, 0 * 4 + (-1) * 5 + 2 * 6]


def test_matrix_vector_product_shape_mismatch():
    A = create_matrix([[1, 2], [3, 4]])
    v = [1, 2, 3]
    with pytest.raises(ValueError):
        matrix_vector_product(A, v)
