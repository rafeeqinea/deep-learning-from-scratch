from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from chapters.ch06_feedforward.api import (
    linear_forward, linear_backward,
    relu_forward, relu_backward,
    glorot_uniform, bias_zeros,
    cross_entropy_from_logits, ce_grad_wrt_logits,
)

Params = Dict[str, np.ndarray]
Cache = Dict[str, Any]

def init_mlp(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0) -> Params:
    rng = np.random.default_rng(seed)
    W1 = glorot_uniform((input_dim, hidden_dim), rng=rng, dtype=np.float64)
    b1 = bias_zeros((hidden_dim,), dtype=np.float64)
    W2 = glorot_uniform((hidden_dim, output_dim), rng=rng, dtype=np.float64)
    b2 = bias_zeros((output_dim,), dtype=np.float64)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_mlp(X: np.ndarray, params: Params) -> Tuple[np.ndarray, Cache]:
    Z1, c_lin1 = linear_forward(X, params["W1"], params["b1"])
    A1, c_relu1 = relu_forward(Z1)
    Z2, c_lin2 = linear_forward(A1, params["W2"], params["b2"])  # logits
    cache = {"lin1": c_lin1, "relu1": c_relu1, "lin2": c_lin2}
    return Z2, cache

def backward_mlp(logits: np.ndarray, targets: np.ndarray, cache: Cache) -> Params:
    dL_dZ2 = ce_grad_wrt_logits(logits, targets)          # (B,C)
    dA1, dW2, db2 = linear_backward(dL_dZ2, cache["lin2"])
    dZ1 = relu_backward(dA1, cache["relu1"])
    dX, dW1, db1 = linear_backward(dZ1, cache["lin1"])
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def ce_loss_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    return cross_entropy_from_logits(logits, targets)
