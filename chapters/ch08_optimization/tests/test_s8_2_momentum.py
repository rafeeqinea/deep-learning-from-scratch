import numpy as np
import pytest
from chapters.ch08_optimization.api import momentum_update

def test_momentum_no_nesterov_matches_formula_dict():
    # v := beta v + g ; step = v ; p -= lr * step
    p = {"w": np.array([1.0, -2.0], dtype=np.float64)}
    g = {"w": np.array([0.5,  1.0], dtype=np.float64)}
    v = None
    lr, beta = 0.1, 0.9

    # step 1: v = g, p = p - lr*v
    v = momentum_update(p, g, v, lr=lr, beta=beta, nesterov=False)
    assert np.allclose(v["w"], g["w"])
    assert np.allclose(p["w"], np.array([1.0, -2.0]) - lr * g["w"])

    # step 2: v = beta*v + g ; p -= lr*v
    p2_before = p["w"].copy()
    v = momentum_update(p, g, v, lr=lr, beta=beta, nesterov=False)
    expected_v = beta * g["w"] + g["w"]
    assert np.allclose(v["w"], expected_v)
    assert np.allclose(p["w"], p2_before - lr * expected_v)

def test_momentum_nesterov_uses_lookahead_step_list():
    # with Nesterov: step = beta*v + g  (v already updated to beta*v + g)
    W = np.array([[1.0]], dtype=np.float64)
    params = [W]
    grads  = [np.array([[2.0]], dtype=np.float64)]
    v = None
    lr, beta = 0.5, 0.9

    # step 1
    v = momentum_update(params, grads, v, lr=lr, beta=beta, nesterov=True)
    # v = g ; step = beta*v + g = (0.9*2 + 2) = 3.8 ; W -= 0.5*3.8 = 1 - 1.9 = -0.9
    assert np.allclose(v[0], np.array([[2.0]]))
    assert np.allclose(params[0], np.array([[-0.9]]))

def test_momentum_invalid_args_and_state_shape():
    params = {"w": np.zeros((2,))}
    grads  = {"w": np.zeros((2,))}
    with pytest.raises(ValueError):
        momentum_update(params, grads, None, lr=0.0)
    with pytest.raises(ValueError):
        momentum_update(params, grads, None, lr=0.1, beta=1.0)

    # bad state structure
    bad_state = [np.zeros((2,))]
    with pytest.raises(ValueError):
        momentum_update(params, grads, bad_state, lr=0.1)

def test_momentum_skips_none_grads_and_is_inplace():
    w = np.array([1.0, 2.0], dtype=np.float64)
    b = np.array([0.5], dtype=np.float64)
    params = {"w": w, "b": b}
    grads  = {"w": np.array([0.1, -0.2]), "b": None}
    wid, bid = id(params["w"]), id(params["b"])

    v = momentum_update(params, grads, None, lr=0.1, beta=0.9, nesterov=False)
    assert id(params["w"]) == wid and id(params["b"]) == bid
    # b unchanged
    assert np.allclose(params["b"], b)
    # w moved opposite grad
    assert np.allclose(params["w"], np.array([1.0, 2.0]) - 0.1 * np.array([0.1, -0.2]))
    # state returned
    assert isinstance(v, dict) and "w" in v and "b" in v
