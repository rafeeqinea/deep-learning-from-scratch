import numpy as np
from chapters.ch03_probability.s3_6 import (
    chain_rule_product_equals_joint_3d
)

def test_chain_rule_matches_joint_random_3d():
    rng = np.random.default_rng(0)
    P = rng.random((2,3,2))
    P /= P.sum()
    # default order
    assert chain_rule_product_equals_joint_3d(P, order=(0,1,2)) is True
    # permuted axes
    assert chain_rule_product_equals_joint_3d(P.transpose(1,2,0), order=(0,1,2)) is True
