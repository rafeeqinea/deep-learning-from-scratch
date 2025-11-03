import numpy as np
from chapters.ch04_numerical.s4_2.conditioning import cond2_via_svd

def test_cond2_matches_numpy_linalg_cond():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    ours = cond2_via_svd(A)
    theirs = np.linalg.cond(A, 2)
    assert np.allclose(ours, theirs, rtol=1e-10, atol=1e-12)

def test_cond2_inf_for_singular():
    A = np.array([[1.0, 2.0],
                  [2.0, 4.0]], dtype=np.float64)  # rank-1
    assert cond2_via_svd(A) == float("inf")
