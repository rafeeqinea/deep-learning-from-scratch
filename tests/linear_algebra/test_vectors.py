import pytest
from core.linear_algebra.vectors import (
    create_vector, add_vectors, subtract_vectors,
    scalar_multiply, dot_product, vector_norm, normalize_vector
)

def test_create_vector_happy():
    v = create_vector([1, 2, 3])
    assert v == [1, 2, 3]

def test_create_vector_empty_raises():
    with pytest.raises(ValueError):
        create_vector([])

def test_create_vector_non_numeric_raises():
    with pytest.raises(TypeError):
        create_vector([1, "a", 3])

def test_add_vectors():
    a = create_vector([1, 2])
    b = create_vector([3, 4])
    assert add_vectors(a, b) == [4, 6]

def test_subtract_vectors():
    a = create_vector([5, 7])
    b = create_vector([2, 4])
    assert subtract_vectors(a, b) == [3, 3]

def test_scalar_multiply():
    v = create_vector([1, -2])
    assert scalar_multiply(3, v) == [3, -6]

def test_scalar_multiply_invalid_scalar():
    with pytest.raises(TypeError):
        scalar_multiply("x", [1, 2])

def test_dot_product():
    a = create_vector([1, 2, 3])
    b = create_vector([4, 5, 6])
    assert dot_product(a, b) == 32

def test_dot_product_length_mismatch():
    with pytest.raises(ValueError):
        dot_product([1, 2], [1])

def test_vector_norm_l1():
    v = create_vector([-1, 2, -3])
    assert vector_norm(v, 1) == 6.0

def test_vector_norm_l2():
    v = create_vector([3, 4])
    assert vector_norm(v, 2) == 5.0

def test_vector_norm_inf():
    v = create_vector([1, -7, 5])
    assert vector_norm(v, "inf") == 7.0

def test_vector_norm_invalid():
    with pytest.raises(ValueError):
        vector_norm([1, 2], 3)

def test_normalize_vector():
    v = create_vector([3, 4])
    normed = normalize_vector(v)
    assert pytest.approx(normed[0]) == 0.6
    assert pytest.approx(normed[1]) == 0.8

def test_normalize_zero_vector():
    with pytest.raises(ValueError):
        normalize_vector([0, 0])
