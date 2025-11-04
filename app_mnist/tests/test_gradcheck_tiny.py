import numpy as np
from app_mnist.model_mlp import init_mlp, forward_mlp, backward_mlp, ce_loss_from_logits
from chapters.ch04_numerical.api import central_diff_grad

def test_tiny_mlp_gradcheck():
    rng = np.random.default_rng(123)
    B, Din, H, C = 2, 4, 3, 3
    X = rng.normal(size=(B, Din)).astype(np.float64)
    y = rng.integers(0, C, size=(B,))

    params = init_mlp(Din, H, C, seed=0)

    # analytic grads
    logits, cache = forward_mlp(X, params)
    grads = backward_mlp(logits, y, cache)

    # numeric grads (flatten/unflatten helper)
    def pack(p):
        return np.concatenate([p["W1"].ravel(), p["b1"].ravel(), p["W2"].ravel(), p["b2"].ravel()])

    shapes = {k: params[k].shape for k in params}
    sizes  = {k: params[k].size for k in params}
    offsets = {}
    off = 0
    for k in ["W1","b1","W2","b2"]:
        offsets[k] = (off, off + sizes[k])
        off += sizes[k]

    def unpack(vec):
        out = {}
        for k in ["W1","b1","W2","b2"]:
            s, e = offsets[k]
            out[k] = vec[s:e].reshape(shapes[k]).copy()
        return out

    def f(vec):
        p = unpack(vec)
        z, _ = forward_mlp(X, p)
        return ce_loss_from_logits(z, y)

    num = central_diff_grad(f, pack(params).copy(), eps=1e-6)
    ana = pack(grads)

    assert np.allclose(ana, num, rtol=5e-5, atol=5e-7)
