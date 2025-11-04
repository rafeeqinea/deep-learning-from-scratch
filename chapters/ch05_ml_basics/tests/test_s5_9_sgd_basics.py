import numpy as np
from chapters.ch05_ml_basics.api import set_all_seeds, num_batches, batch_iterator

def test_set_all_seeds_determinism_numpy_random():
    set_all_seeds(42)
    a = np.random.rand(5)
    set_all_seeds(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)

def test_num_batches_drop_last_and_keep():
    assert num_batches(10, 4, drop_last=False) == 3  # 4+4+2
    assert num_batches(10, 4, drop_last=True)  == 2

def test_batch_iterator_shapes_and_order_no_shuffle():
    X = np.arange(10).reshape(10,1)
    y = np.arange(10)
    batches = list(batch_iterator(X, y, batch_size=4, shuffle=False, drop_last=False))
    lens = [xb.shape[0] for xb, _ in batches]
    assert lens == [4,4,2]
    # first batch should be 0..3
    assert np.array_equal(batches[0][0].ravel(), np.array([0,1,2,3]))

def test_batch_iterator_shuffle_with_seed_is_deterministic():
    X = np.arange(10).reshape(10,1)
    y = np.arange(10)
    b1 = list(batch_iterator(X, y, batch_size=3, shuffle=True, seed=7))
    b2 = list(batch_iterator(X, y, batch_size=3, shuffle=True, seed=7))
    # concatenated indices equal
    idx1 = np.concatenate([xb.ravel() for xb, _ in b1])
    idx2 = np.concatenate([xb.ravel() for xb, _ in b2])
    assert np.array_equal(idx1, idx2)

def test_batch_iterator_y_none():
    X = np.arange(5).reshape(5,1)
    batches = list(batch_iterator(X, None, batch_size=2, shuffle=False))
    # yields (Xb, None)
    assert batches[0][1] is None
