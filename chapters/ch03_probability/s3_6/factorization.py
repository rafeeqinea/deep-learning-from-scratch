# Chain rule utilities for small discrete tables (NumPy only).
import numpy as np

def _check_joint_nd(P):
    P = np.asarray(P, dtype=float)
    if np.any(P < -1e-12):
        raise ValueError("joint has negative entries.")
    s = P.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("joint must have positive finite sum.")
    P = P / s
    return P

def marginalize_nd(P, axes_to_keep):
    """
    Sum out all axes not in 'axes_to_keep' from an ND joint table.
    axes_to_keep: tuple/list of ints in sorted order.
    Returns the marginalized table with those axes preserved.
    """
    P = _check_joint_nd(P)
    axes = tuple(range(P.ndim))
    axes_to_keep = tuple(int(a) for a in axes_to_keep)
    axes_to_sum = tuple(a for a in axes if a not in axes_to_keep)
    if len(axes_to_sum) == 0:
        return P
    return P.sum(axis=axes_to_sum)

def _conditional_from_joint_2d(P2, given_axis):
    # helper like s3_5.cond_pmf but without re-importing
    if given_axis == 1:
        s = P2.sum(axis=0, keepdims=True)
        if np.any(s == 0): raise ValueError("zero-prob conditional")
        return P2 / s
    elif given_axis == 0:
        s = P2.sum(axis=1, keepdims=True)
        if np.any(s == 0): raise ValueError("zero-prob conditional")
        return (P2 / s).T
    else:
        raise ValueError

def chain_rule_product_equals_joint_3d(Pabc, order=(0,1,2), atol=1e-12):
    """
    Verify chain rule on a 3D joint P(A,B,C):
      P(a,b,c) = P(a) * P(b|a) * P(c|a,b)   (for order=(0,1,2))
    Supports any permutation 'order' of axes.

    Returns True if the product matches the original joint within atol.
    """
    P = _check_joint_nd(Pabc)
    if P.ndim != 3:
        raise ValueError("expect 3-D joint table.")
    a,b,c = order  # permutation of (0,1,2)

    # P(a)
    Pa = marginalize_nd(P, (a,))
    # P(b|a): for each a, conditional distribution over b
    Pab = marginalize_nd(P, (a,b))                   # shape (|a|, |b|)
    Pb_given_a = _conditional_from_joint_2d(Pab, given_axis=0)  # (|b|, |a|)

    # P(c|a,b): for every (a,b) slice, normalize over c
    Pabc_norm = P / (marginalize_nd(P, (a,b)).reshape(Pab.shape + (1,)) + 1e-20)
    # Now multiply: P(a) * P(b|a) * P(c|a,b)
    # Broadcast shapes carefully
    # Pa: (|a|,) -> reshape to put at axis 'a'
    Pa_full = np.expand_dims(Pa, axis=(1 if a==0 else 0))
    # Reorder Pb|a into an array aligned with axes (a,b)
    Pbga = Pb_given_a.T  # (|a|, |b|)
    prod = (Pa_full * Pbga)[(...,)]  # shape (|a|, |b|)
    # Now expand over c and multiply by P(c|a,b)
    prod = np.expand_dims(prod, axis=2) * Pabc_norm  # (|a|, |b|, |c|)

    # Reorder prod axes back to (0,1,2) to compare with P
    if order != (0,1,2):
        # Compute permutation to move axes (a,b,c) -> (0,1,2)
        inv = np.argsort(order)
        prod = np.transpose(prod, axes=tuple(inv))
    return bool(np.allclose(prod, P, atol=atol))
