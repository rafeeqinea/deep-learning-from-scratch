import numpy as np
from typing import Dict, Iterable, Tuple, Union, Optional, Any

Array = np.ndarray
ParamStruct = Union[Dict[str, Array], Iterable[Array]]

def _iter_struct(x: ParamStruct) -> Iterable[Tuple[str, Any]]:
    if isinstance(x, dict):
        for k, v in x.items():
            yield k, v
    else:
        for i, v in enumerate(x):
            yield f"p{i}", v

def _ensure_match(params: ParamStruct, grads: ParamStruct) -> None:
    if isinstance(params, dict) and isinstance(grads, dict):
        if set(params.keys()) != set(grads.keys()):
            raise ValueError("params and grads dicts must have same keys")
        for k in params:
            g = grads[k]
            if g is None:  # allow None gradients
                continue
            if np.shape(params[k]) != np.shape(g):
                raise ValueError(f"shape mismatch for key '{k}'")
    elif not isinstance(params, dict) and not isinstance(grads, dict):
        p_list = list(_iter_struct(params))
        g_list = list(_iter_struct(grads))
        if len(p_list) != len(g_list):
            raise ValueError("params and grads lists must have same length")
        for (kp, p), (kg, g) in zip(p_list, g_list):
            if g is None:  # allow None gradients
                continue
            if np.shape(p) != np.shape(g):
                raise ValueError(f"shape mismatch at index {kp}")
    else:
        raise ValueError("params and grads must both be dict or both be list/tuple")

def sgd_update(params: ParamStruct,
               grads: ParamStruct,
               lr: float) -> None:
    """
    In-place SGD update: param -= lr * grad
    - Works with dict or list/tuple structures.
    - Allows None gradient entries (ignored).
    """
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be positive and finite")

    _ensure_match(params, grads)

    if isinstance(params, dict):
        for k, p in params.items():
            g = grads[k]
            if g is None:
                continue
            params[k][...] = np.asarray(p) - float(lr) * np.asarray(g, dtype=p.dtype)
    else:
        for i, (name, p) in enumerate(_iter_struct(params)):
            g = list(grads)[i] if not isinstance(grads, dict) else grads[name]
            if g is None:
                continue
            params[i][...] = np.asarray(p) - float(lr) * np.asarray(g, dtype=p.dtype)
