import numpy as np
from typing import Dict, Iterable, Tuple, Union

Array = np.ndarray
ParamStruct = Union[Dict[str, Array], Iterable[Array]]

def _validate_lambda(lmbda: float) -> None:
    if not np.isfinite(lmbda):
        raise ValueError("lambda must be finite")
    if lmbda < 0.0:
        raise ValueError("lambda must be >= 0")

def _is_bias(name: str, p: Array) -> bool:
    """Heuristic: 1D tensors or names like 'b' / 'bias' are considered biases."""
    if p.ndim == 1:
        return True
    n = name.lower()
    return n.endswith("b") or "bias" in n

def _iter_params(params: ParamStruct) -> Iterable[Tuple[str, Array]]:
    """Yield (name, array) regardless of dict/list input."""
    if isinstance(params, dict):
        for k, v in params.items():
            yield k, np.asarray(v)
    else:
        # list/tuple of arrays -> synthesize names 'p0', 'p1', ...
        for i, v in enumerate(params):
            yield f"p{i}", np.asarray(v)

def l2_value(params: ParamStruct,
             lmbda: float,
             exclude_biases: bool = True) -> float:
    """
    L2 regularization value (a.k.a. weight decay penalty):
        (λ/2) * Σ ||W||²
    If exclude_biases=True, 1D tensors (biases) and names like '*b' or '*bias'
    are excluded from the sum.
    Returns a Python float.
    """
    _validate_lambda(lmbda)
    total = 0.0
    for name, p in _iter_params(params):
        if exclude_biases and _is_bias(name, p):
            continue
        total += float(np.sum(p.astype(np.float64) ** 2))
    return 0.5 * lmbda * total

def l2_grad(params: ParamStruct,
            lmbda: float,
            exclude_biases: bool = True) -> ParamStruct:
    """
    Gradient of L2 value wrt params:
        ∂/∂W ( (λ/2) ||W||² ) = λ W
    For excluded biases, gradient is zeros_like(bias).
    Returns same structure type as input (dict or list).
    """
    _validate_lambda(lmbda)

    if isinstance(params, dict):
        grads: Dict[str, Array] = {}
        for name, p in _iter_params(params):
            if exclude_biases and _is_bias(name, p):
                grads[name] = np.zeros_like(p, dtype=np.float64)
            else:
                grads[name] = (lmbda * p.astype(np.float64))
        return grads

    # list/tuple case: keep order
    outs = []
    for name, p in _iter_params(params):
        if exclude_biases and _is_bias(name, p):
            outs.append(np.zeros_like(p, dtype=np.float64))
        else:
            outs.append(lmbda * p.astype(np.float64))
    return outs

def apply_weight_decay(param_grads: ParamStruct,
                       params: ParamStruct,
                       lmbda: float,
                       exclude_biases: bool = True) -> ParamStruct:
    """
    Add L2 gradient (λW) into existing param_grads structure, in-place shape.
    If param_grads is dict/list matching params, returns the same structure.
    Useful right before optimizer step:
        grads = backward(...)
        grads = apply_weight_decay(grads, params, λ, exclude_biases=True)
        sgd_update(params, grads, lr)
    """
    _validate_lambda(lmbda)

    if isinstance(params, dict):
        if not isinstance(param_grads, dict):
            raise ValueError("param_grads must be a dict to match dict params")
        for name, p in _iter_params(params):
            wd = np.zeros_like(p, dtype=np.float64) if (exclude_biases and _is_bias(name, p)) else (lmbda * p.astype(np.float64))
            if name not in param_grads:
                param_grads[name] = wd
            else:
                param_grads[name] = np.asarray(param_grads[name], dtype=np.float64) + wd
        return param_grads

    # list/tuple case
    if not isinstance(param_grads, (list, tuple)):
        raise ValueError("param_grads must be list/tuple to match list/tuple params")

    out = []
    for (name, p), g in zip(_iter_params(params), param_grads):
        wd = np.zeros_like(p, dtype=np.float64) if (exclude_biases and _is_bias(name, p)) else (lmbda * p.astype(np.float64))
        out.append(np.asarray(g, dtype=np.float64) + wd)
    return out
