import numpy as np
from typing import Dict, Iterable, Tuple, Union, Optional

Array = np.ndarray
ParamStruct = Union[Dict[str, Array], Iterable[Array]]

def _iter_struct(x: ParamStruct):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k, np.asarray(v)
    else:
        for i, v in enumerate(x):
            yield f"p{i}", np.asarray(v)

def _zeros_like_structure(params: ParamStruct) -> ParamStruct:
    if isinstance(params, dict):
        return {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
    else:
        return [np.zeros_like(v, dtype=np.float64) for _, v in _iter_struct(params)]

def _match_struct(a: ParamStruct, b: ParamStruct) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys())
    if not isinstance(a, dict) and not isinstance(b, dict):
        return len(list(_iter_struct(a))) == len(list(_iter_struct(b)))
    return False

def momentum_update(params: ParamStruct,
                    grads: ParamStruct,
                    state: Optional[ParamStruct],
                    lr: float,
                    beta: float = 0.9,
                    nesterov: bool = False) -> ParamStruct:
    """
    Polyak momentum (optionally Nesterov), in-place param update.
    Convention used:
        v := beta * v + grad
        if nesterov: step = beta * v + grad
        else:       step = v
        param -= lr * step
    Returns the updated velocity state 'v' with same structure as params.
    - Allows None gradients (skipped).
    """
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be positive and finite")
    if not np.isfinite(beta) or not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0,1)")

    # init or validate state
    if state is None:
        v = _zeros_like_structure(params)
    else:
        if not _match_struct(params, state):
            raise ValueError("state structure must match params")
        v = state

    if isinstance(params, dict):
        if not isinstance(grads, dict):
            raise ValueError("grads must be dict to match dict params")
        for k, p in params.items():
            g = grads.get(k, None)
            if g is None:
                continue
            v[k][...] = beta * v[k] + np.asarray(g, dtype=np.float64)
            step = (beta * v[k] + np.asarray(g, dtype=np.float64)) if nesterov else v[k]
            params[k][...] = np.asarray(p) - float(lr) * step.astype(p.dtype, copy=False)
    else:
        if isinstance(grads, dict):
            raise ValueError("grads must be list/tuple to match list/tuple params")
        p_list = list(_iter_struct(params))
        g_list = list(_iter_struct(grads))
        for i, ((_, p), (_, g)) in enumerate(zip(p_list, g_list)):
            if g is None:
                continue
            v[i][...] = beta * v[i] + np.asarray(g, dtype=np.float64)
            step = (beta * v[i] + np.asarray(g, dtype=np.float64)) if nesterov else v[i]
            params[i][...] = np.asarray(p) - float(lr) * step.astype(p.dtype, copy=False)

    return v
