import numpy as np
from chapters.ch04_numerical.s4_3.grad_theory import (
    make_ls_quadratic, lipschitz_const_from_A, descent_lemma_gap
)

def test_make_ls_quadratic_grad_matches_formula():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 4))
    b = rng.normal(size=(6,))
    f, g = make_ls_quadratic(A, b)

    x = rng.normal(size=(4,))
    # grad should be A^T (A x - b)
    expected = A.T @ (A @ x - b)
    assert np.allclose(g(x), expected, rtol=1e-12, atol=1e-12)
    # f is scalar
    assert isinstance(f(x), float)

def test_lipschitz_and_descent_lemma_gap_nonpositive():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    b = rng.normal(size=(5,))
    f, g = make_ls_quadratic(A, b)
    L = lipschitz_const_from_A(A)
    x = rng.normal(size=(5,))
    y = rng.normal(size=(5,))
    gap = descent_lemma_gap(f, g, x, y, L)
    # Numerically should be <= ~1e-10 (allow tiny positive epsilon)
    assert gap <= 1e-10
