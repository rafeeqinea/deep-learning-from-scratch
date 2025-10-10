# Conditional probability utilities (NumPy only).
import numpy as np

def _check_probs(p, name="p", atol=1e-8):
    p = np.asarray(p, dtype=float)
    if np.any(p < -1e-12):
        raise ValueError(f"{name} has negative entries.")
    s = float(p.sum())
    if not np.isfinite(s) or abs(s - 1.0) > atol:
        raise ValueError(f"{name} must sum to 1 (got {s}).")
    return p

def pmf_from_counts(counts, axis=None):
    """
    Normalize nonnegative counts to a probability mass function.

    counts : array-like of nonnegatives
    axis   : if None, normalize all; else normalize along 'axis'
    """
    C = np.asarray(counts, dtype=float)
    if np.any(C < 0):
        raise ValueError("counts must be nonnegative.")
    if axis is None:
        Z = C.sum()
        if Z == 0:
            raise ValueError("counts sum to zero.")
        return C / Z
    Z = C.sum(axis=axis, keepdims=True)
    if np.any(Z == 0):
        raise ValueError("some slices have zero total count.")
    return C / Z

def marginal_from_joint(joint, axis):
    """
    Marginalize a joint table by summing over 'axis'.

    joint: ndarray with nonnegative entries that sum to 1
    axis : int (which axis to sum out)
    """
    J = _check_probs(joint, name="joint")
    return J.sum(axis=axis)

def cond_pmf(joint, given_axis):
    """
    Conditional table from a joint P.

    For a 2D joint P(X,Y) with shape (n_x, n_y):

      given_axis = 1 (given Y):
          returns P(X|Y): shape (n_x, n_y) where each column sums to 1

      given_axis = 0 (given X):
          returns P(Y|X): shape (n_y, n_x) where each column sums to 1

    For ND > 2:
      returns a table normalized along the 'given_axis' complement so that
      for each fixed value of the given variable(s), the conditional over the
      remaining variable sums to 1 along that axis.
    """
    P = _check_probs(joint, name="joint")
    if P.ndim != 2:
        raise ValueError("cond_pmf: expected a 2-D joint table.")
    if given_axis not in (0, 1):
        raise ValueError("cond_pmf: given_axis must be 0 or 1 for 2-D tables.")

    if given_axis == 1:
        # columns = fixed Y; normalize each column to get P(X|Y)
        col_sums = P.sum(axis=0, keepdims=True)  # shape (1, n_y)
        if np.any(col_sums == 0):
            raise ValueError("Some P(Y=y) are zero; cannot condition on zero-prob events.")
        return P / col_sums  # shape (n_x, n_y)
    else:
        # rows = fixed X; normalize each row to get P(Y|X), return as (n_y, n_x)
        row_sums = P.sum(axis=1, keepdims=True)  # (n_x, 1)
        if np.any(row_sums == 0):
            raise ValueError("Some P(X=x) are zero; cannot condition on zero-prob events.")
        return (P / row_sums).T  # (n_y, n_x): columns correspond to X

def joint_from_cond_and_prior(cond, prior, given_axis):
    """
    Reconstruct joint from conditional and prior.

    For joint P(X,Y), if given_axis=1 (given Y):
       cond = P(X|Y) shape (n_x, n_y)
       prior = P(Y) shape (n_y,)
       joint[i,j] = cond[i,j] * prior[j]

    If given_axis=0 (given X):
       cond = P(Y|X) shape (n_y, n_x)
       prior = P(X) shape (n_x,)
       joint[i,j] = cond[j,i] * prior[i]
    """
    cond = np.asarray(cond, dtype=float)
    prior = _check_probs(prior, name="prior")
    if cond.ndim != 2:
        raise ValueError("cond must be 2-D.")
    if given_axis == 1:
        n_x, n_y = cond.shape
        if prior.shape != (n_y,):
            raise ValueError("prior must have shape (n_y,) for given_axis=1.")
        joint = cond * prior.reshape(1, n_y)
    elif given_axis == 0:
        n_y, n_x = cond.shape
        if prior.shape != (n_x,):
            raise ValueError("prior must have shape (n_x,) for given_axis=0.")
        # joint (n_x, n_y)
        joint = (cond.T) * prior.reshape(n_x, 1)
    else:
        raise ValueError("given_axis must be 0 or 1.")
    # Renormalize tiny drift
    joint = np.maximum(joint, 0.0)
    joint /= joint.sum()
    return joint

def law_total_probability_check(joint, given_axis, atol=1e-10):
    """
    Check law of total probability on a 2-D joint P(X,Y):
      P(X) = sum_y P(X|Y=y) P(Y=y)
      or P(Y) = sum_x P(Y|X=x) P(X=x)

    Returns True if both sides match within atol.
    """
    P = _check_probs(joint, name="joint")
    if P.ndim != 2:
        raise ValueError("expect 2-D joint.")
    if given_axis == 1:
        px = P.sum(axis=1)           # P(X)
        py = P.sum(axis=0)           # P(Y)
        p_x_given_y = cond_pmf(P, given_axis=1)  # (n_x, n_y)
        px_from_total = (p_x_given_y * py.reshape(1, -1)).sum(axis=1)
        return bool(np.allclose(px_from_total, px, atol=atol))
    elif given_axis == 0:
        px = P.sum(axis=1)           # P(X)
        py = P.sum(axis=0)           # P(Y)
        p_y_given_x = cond_pmf(P, given_axis=0)  # (n_y, n_x)
        py_from_total = (p_y_given_x * px.reshape(1, -1)).sum(axis=1)
        return bool(np.allclose(py_from_total, py, atol=atol))
    else:
        raise ValueError("given_axis must be 0 or 1.")
