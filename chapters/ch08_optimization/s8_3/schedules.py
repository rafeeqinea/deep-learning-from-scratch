import numpy as np

def cosine_lr(step: int,
              total_steps: int,
              base_lr: float,
              min_lr: float = 0.0,
              warmup_steps: int = 0) -> float:
    """
    Cosine schedule with optional linear warmup.
      warmup: lr = base_lr * step / warmup_steps   for step < warmup_steps
      cosine: min_lr + 0.5*(base_lr-min_lr)*(1+cos(pi*t)),  t in [0,1]
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if step < 0:
        step = 0
    if step > total_steps:
        step = total_steps
    if not np.isfinite(base_lr) or base_lr <= 0.0:
        raise ValueError("base_lr must be positive and finite")
    if min_lr < 0 or min_lr > base_lr:
        raise ValueError("min_lr must be in [0, base_lr]")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")

    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr * (step / float(max(1, warmup_steps))))

    # cosine phase
    # map step from [warmup_steps, total_steps] -> t in [0,1]
    denom = max(1, total_steps - warmup_steps)
    t = (step - warmup_steps) / denom
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * t)))
