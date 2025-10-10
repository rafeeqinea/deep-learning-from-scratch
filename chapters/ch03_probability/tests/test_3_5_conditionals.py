import numpy as np
from chapters.ch03_probability.s3_5 import (
    pmf_from_counts, marginal_from_joint, cond_pmf,
    joint_from_cond_and_prior, law_total_probability_check
)

def test_pmf_and_conditionals_and_total_prob():
    # Joint over X={0,1}, Y={0,1,2}
    counts = np.array([[10, 5, 0],
                       [10, 5, 10]], dtype=float)
    joint = pmf_from_counts(counts)   # normalize over all
    assert np.isclose(joint.sum(), 1.0)

    px = marginal_from_joint(joint, axis=1)  # sum over Y -> P(X)
    py = marginal_from_joint(joint, axis=0)  # sum over X -> P(Y)
    assert np.isclose(px.sum(), 1.0) and np.isclose(py.sum(), 1.0)

    # P(X|Y)
    px_given_y = cond_pmf(joint, given_axis=1)
    # each column sums to 1 where P(Y)>0
    col_sums = px_given_y.sum(axis=0)
    # column 2 had zero counts for X=0, but not for X=1 => still OK
    assert np.allclose(col_sums[py > 0], 1.0)

    # Reconstruct joint from conditional and prior
    joint2 = joint_from_cond_and_prior(px_given_y, py, given_axis=1)
    assert np.allclose(joint2, joint, atol=1e-12)

    # Law of total probability
    assert law_total_probability_check(joint, given_axis=1) is True
    assert law_total_probability_check(joint, given_axis=0) is True
