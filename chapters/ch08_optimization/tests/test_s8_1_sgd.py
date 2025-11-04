import numpy as np
import pytest
from chapters.ch08_optimization.api import sgd_update

def test_sgd_update_dict_inplace_and_values():
    W = np.array([[1.0, -2.0]], dtype=np.float32)
    b = np.array([0.5], dtype=np.float32)
    params = {"W": W, "b": b}
    grads  = {"W": np.array([[0.1, 0.2]], dtype=np.float32), "b": np.array([0.05], dtype=np.float32)}

    idW_before, idb_before = id(params["W"]), id(params["b"])
    sgd_update(params, grads, lr=0.1)
    # in-place: same objects
    assert id(params["W"]) == idW_before
    assert id(params["b"]) == idb_before
    # values: p -= lr*g
    assert np.allclose(params["W"], np.array([[0.99, -2.02]], dtype=np.float32))
    assert np.allclose(params["b"], np.array([0.495], dtype=np.float32))

def test_sgd_update_list_inplace_and_values_with_none_grad():
    W = np.array([1.0, -2.0], dtype=np.float64)
    b = np.array([0.5], dtype=np.float64)
    params = [W, b]
    grads  = [np.array([0.1, 0.2], dtype=np.float64), None]  # b has None grad

    idW, idb = id(params[0]), id(params[1])
    sgd_update(params, grads, lr=0.5)
    assert id(params[0]) == idW and id(params[1]) == idb  # in-place
    # W updated, b unchanged
    assert np.allclose(params[0], np.array([0.95, -2.10]))
    assert np.allclose(params[1], np.array([0.5]))

def test_sgd_update_key_mismatch_raises():
    params = {"W": np.zeros((2,2))}
    grads  = {"U": np.zeros((2,2))}
    with pytest.raises(ValueError):
        sgd_update(params, grads, lr=0.1)

def test_sgd_update_shape_mismatch_raises():
    params = {"W": np.zeros((2,2))}
    grads  = {"W": np.zeros((3,))}
    with pytest.raises(ValueError):
        sgd_update(params, grads, lr=0.1)

def test_sgd_update_invalid_lr_raises():
    params = {"W": np.zeros((1,))}
    grads  = {"W": np.zeros((1,))}
    with pytest.raises(ValueError):
        sgd_update(params, grads, lr=0.0)
    with pytest.raises(ValueError):
        sgd_update(params, grads, lr=float("inf"))
