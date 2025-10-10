# Likelihood / NLL / MAP utilities (NumPy only).
import numpy as np

from chapters.ch03_probability.s3_10.prob_numerics import (
    log_softmax, cross_entropy_from_logits, binary_cross_entropy_from_logits
)

def nll_categorical_from_logits(logits, y, reduction="mean"):
    """
    Negative log-likelihood for multiclass classification from unnormalized logits.
      logits : (batch, C)
      y      : class indices (batch,) or one-hot (batch, C)

    Returns scalar if reduction="mean"/"sum", else per-example vector.
    """
    ce = cross_entropy_from_logits(logits, y, reduction="none")
    if reduction == "none":
        return ce
    if reduction == "sum":
        return float(ce.sum())
    return float(ce.mean())

def nll_binary_from_logits(logits, y, reduction="mean"):
    """
    Binary negative log-likelihood (logistic loss) from logits.
      logits : (batch,) or (batch, 1)
      y      : 0/1 targets broadcastable to logits
    """
    bce = binary_cross_entropy_from_logits(logits, y, reduction="none")
    if reduction == "none":
        return bce
    if reduction == "sum":
        return float(bce.sum())
    return float(bce.mean())

def l2_penalty(W, lam):
    """
    1/2 * lam * ||W||_2^2
    """
    W = np.asarray(W, dtype=float)
    return 0.5 * float(lam) * float(np.sum(W * W))

def nll_with_l2(nll_value, weights, lam):
    """
    Add L2 penalty to an NLL value (scalar), modeling a Gaussian prior in MAP.
    """
    return float(nll_value) + l2_penalty(weights, lam)
