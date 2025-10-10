import numpy as np
from chapters.ch03_probability.s3_8 import entropy, cross_entropy, kl_divergence

def test_entropy_ce_kl_relationships():
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.4, 0.4, 0.2])

    H = entropy(p)
    CE = cross_entropy(p, q)
    KL = kl_divergence(p, q)

    assert H >= 0 and KL >= 0
    # CE = H + KL
    assert np.isclose(CE, H + KL)
