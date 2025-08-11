# tests/linear_algebra/test_matrix_matrix_product.py
import pytest
from core.linear_algebra.matrix_matrix_product import matrix_product
from core.linear_algebra.matrices import create_matrix

def test_matrix_product_happy_path():
    """Tests the product of two compatible rectangular matrices."""
    A = create_matrix([[1, 2, 3], 
                       [4, 5, 6]])  # 2x3 matrix
    
    B = create_matrix([[7, 8], 
                       [9, 10], 
                       [11, 12]]) # 3x2 matrix
    
    # Expected result is a 2x2 matrix
    C = matrix_product(A, B)
    
    # Row 1 of C: [dot(A_row1, B_col1), dot(A_row1, B_col2)]
    # C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    # C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    # Row 2 of C: [dot(A_row2, B_col1), dot(A_row2, B_col2)]
    # C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    # C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    
    assert C == [[58, 64], [139, 154]]

def test_matrix_product_square():
    """Tests the product of two square matrices."""
    A = create_matrix([[1, 2], [3, 4]])
    B = create_matrix([[5, 6], [7, 8]])
    
    C = matrix_product(A, B)
    
    # C[0][0] = 1*5 + 2*7 = 19
    # C[0][1] = 1*6 + 2*8 = 22
    # C[1][0] = 3*5 + 4*7 = 15 + 28 = 43
    # C[1][1] = 3*6 + 4*8 = 18 + 32 = 50
    assert C == [[19, 22], [43, 50]]

def test_matrix_product_shape_mismatch():
    """Tests that multiplying incompatible matrices raises a ValueError."""
    A = create_matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
    B = create_matrix([[1, 2], [3, 4]])        # 2x2
    
    with pytest.raises(ValueError, match="A has 3 columns but B has 2 rows"):
        matrix_product(A, B)