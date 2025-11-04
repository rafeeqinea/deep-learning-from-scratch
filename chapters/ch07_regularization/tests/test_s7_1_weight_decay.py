import numpy as np
import pytest
from chapters.ch07_regularization.api import l2_value, l2_grad, apply_weight_decay

def test_l2_value_excludes_biases():
    W = np.array([[1.0, -2.0],[3.0, -4.0]], dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    params = {"W": W, "b": b}
    lam = 0.1
    # (λ/2) * (1+4+9+16) = 0.05 * 30 = 1.5
    assert np.isclose(l2_value(params, lam, exclude_biases=True), 1.5)

def test_l2_value_includes_biases():
    W = np.array([[1.0, -2.0],[3.0, -4.0]], dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    params = {"W": W, "b": b}
    lam = 0.1
    # add biases: 0.5^2 + (-0.5)^2 = 0.5 -> total 30.5
    # (λ/2) * 30.5 = 0.05 * 30.5 = 1.525
    assert np.isclose(l2_value(params, lam, exclude_biases=False), 1.525)

def test_l2_grad_excludes_biases():
    W = np.array([[1.0, -2.0],[3.0, -4.0]], dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    params = {"W": W, "b": b}
    lam = 0.1
    grads = l2_grad(params, lam, exclude_biases=True)
    assert np.allclose(grads["W"], lam * W)
    assert np.allclose(grads["b"], np.zeros_like(b))

def test_l2_grad_includes_biases():
    W = np.array([[1.0, -2.0],[3.0, -4.0]], dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    params = {"W": W, "b": b}
    lam = 0.1
    grads = l2_grad(params, lam, exclude_biases=False)
    assert np.allclose(grads["W"], lam * W)
    assert np.allclose(grads["b"], lam * b)

def test_apply_weight_decay_dict_structure_and_values():
    W = np.array([[ 2.0, -1.0],[ 0.5, 3.0]], dtype=np.float64)
    b = np.array([0.1, -0.2], dtype=np.float64)
    params = {"W": W, "b": b}
    grads  = {"W": np.ones_like(W), "b": np.ones_like(b)}  # pretend from backprop
    lam = 0.2

    out = apply_weight_decay(grads, params, lam, exclude_biases=True)
    # W grad += λW ; b unchanged (excluded)
    assert np.allclose(out["W"], np.ones_like(W) + lam * W)
    assert np.allclose(out["b"], np.ones_like(b))  # unchanged

def test_apply_weight_decay_list_structure_and_values():
    W = np.array([[1.0, 2.0]], dtype=np.float64)
    b = np.array([0.3], dtype=np.float64)
    params = [W, b]
    grads  = [np.zeros_like(W), np.zeros_like(b)]
    lam = 0.5

    out = apply_weight_decay(grads, params, lam, exclude_biases=True)
    assert isinstance(out, list) and len(out) == 2
    assert np.allclose(out[0], lam * W)      # W gets λW
    assert np.allclose(out[1], np.zeros_like(b))  # bias excluded

def test_lambda_validation():
    W = np.zeros((1,1))
    with pytest.raises(ValueError):
        l2_value({"W": W}, -1.0)
    with pytest.raises(ValueError):
        l2_grad({"W": W}, float("nan"))
