import numpy as np
import pytest
from chapters.ch08_optimization.api import cosine_lr

def test_cosine_lr_warmup_and_endpoints():
    base_lr = 0.1
    min_lr  = 0.001
    total   = 100
    warm    = 10

    # warmup is linear from 0 -> base_lr
    assert np.isclose(cosine_lr(0, total, base_lr, min_lr, warm), 0.0)
    assert np.isclose(cosine_lr(warm-1, total, base_lr, min_lr, warm), base_lr * (warm-1)/warm)

    # boundary continuity at warmup -> cosine start equals base_lr
    assert np.isclose(cosine_lr(warm, total, base_lr, min_lr, warm), base_lr)

    # final step equals min_lr
    assert np.isclose(cosine_lr(total, total, base_lr, min_lr, warm), min_lr)

def test_cosine_lr_nonincreasing_after_warmup():
    base_lr = 0.2
    total   = 50
    warm    = 5
    vals = [cosine_lr(s, total, base_lr, min_lr=0.0, warmup_steps=warm) for s in range(warm, total+1)]
    # cosine part should not increase (allow tiny fp jitter)
    diffs = np.diff(vals)
    assert np.all(diffs <= 1e-12)

def test_cosine_lr_invalid_args():
    with pytest.raises(ValueError):
        cosine_lr(step=0, total_steps=0, base_lr=0.1)
    with pytest.raises(ValueError):
        cosine_lr(step=0, total_steps=10, base_lr=0.0)
    with pytest.raises(ValueError):
        cosine_lr(step=0, total_steps=10, base_lr=0.1, min_lr=0.2)
    with pytest.raises(ValueError):
        cosine_lr(step=0, total_steps=10, base_lr=0.1, warmup_steps=-1)
