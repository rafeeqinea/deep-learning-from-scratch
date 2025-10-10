# Stable numerics for probability computations (NumPy only).
import numpy as np

def logsumexp(x, axis=None, keepdims=False):
    """
    log(sum(exp(x))) computed stably by subtracting the max.

    Works along any axis (or over all elements if axis=None).
    """
    x = np.asarray(x, dtype=float)
    if axis is None:
        m = np.max(x)
        z = x - m
        return float(m + np.log(np.sum(np.exp(z))))
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    s = np.sum(np.exp(z), axis=axis, keepdims=True)
    out = m + np.log(s)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def softmax(logits, axis=-1):
    """
    Stable softmax along 'axis'. Returns probabilities that sum to 1 along axis.
    """
    L = np.asarray(logits, dtype=float)
    m = np.max(L, axis=axis, keepdims=True)
    z = L - m
    e = np.exp(z)
    s = e.sum(axis=axis, keepdims=True)
    return e / s

def log_softmax(logits, axis=-1):
    """
    Stable log-softmax along 'axis'.
    """
    L = np.asarray(logits, dtype=float)
    return L - logsumexp(L, axis=axis, keepdims=True)

def one_hot(y, num_classes, dtype=np.float32):
    """
    One-hot encode integer class indices y into shape (batch, C).
    """
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    n = y.shape[0]
    C = int(num_classes)
    out = np.zeros((n, C), dtype=dtype)
    if n == 0:
        return out
    out[np.arange(n), y.astype(int)] = 1
    return out

def cross_entropy_from_logits(logits, targets, reduction="mean"):
    """
    Cross-entropy for multiclass from logits (stable).

      logits  : (batch, C)
      targets : class indices (batch,) or one-hot (batch, C)

    Returns:
      - "none": per-example vector, shape (batch,)
      - "sum" : scalar
      - "mean": scalar (default)
    """
    L = np.asarray(logits, dtype=float)
    if L.ndim != 2:
        raise ValueError("logits must be (batch, C).")
    b, C = L.shape
    if isinstance(targets, np.ndarray) and targets.ndim == 2:
        # one-hot
        Y = targets.astype(float)
        if Y.shape != (b, C):
            raise ValueError("one-hot targets must be (batch, C).")
        logp = log_softmax(L, axis=1)                  # (b, C)
        loss = -(Y * logp).sum(axis=1)                 # (b,)
    else:
        # indices
        y = np.asarray(targets).reshape(-1)
        if y.shape[0] != b:
            raise ValueError("index targets must have length batch.")
        logp = log_softmax(L, axis=1)                  # (b, C)
        loss = -logp[np.arange(b), y.astype(int)]      # (b,)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(loss.sum())
    return float(loss.mean())

def log_sigmoid(x):
    """
    Stable log-sigmoid:
      log σ(x) = -softplus(-x)
    """
    x = np.asarray(x, dtype=float)
    # softplus(t) = log(1 + exp(t)) with stability tricks
    t = -x
    # for large positive t, softplus(t) ≈ t; compute piecewise
    out = np.empty_like(t)
    # threshold ~20 is fine in double precision
    mask = t > 20
    out[mask] = t[mask]
    out[~mask] = np.log1p(np.exp(t[~mask]))
    return -out

def sigmoid(x):
    """
    Numerically stable sigmoid.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def binary_cross_entropy_from_logits(logits, targets, reduction="mean"):
    """
    Binary cross-entropy from logits (stable).
      logits : (batch,) or (batch,1) or any shape broadcastable to targets
      targets: same shape (0/1 floats)

    Uses:
      -log σ(z)   when y=1
      -log (1-σ(z)) = -log σ(-z) when y=0
    Combined: softplus(z) - y*z
    """
    z = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=float)
    # Squeeze logits if shape is (batch, 1) to avoid incorrect broadcasting
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(axis=1)
    # Broadcast
    z, y = np.broadcast_arrays(z, y)
    # softplus(z) with stability
    sp = np.empty_like(z)
    mask = z > 20
    sp[mask] = z[mask]                  # softplus ≈ z for large positive
    sp[~mask] = np.log1p(np.exp(z[~mask]))
    loss = sp - y * z
    # reduction
    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(loss.sum())
    return float(loss.mean())
