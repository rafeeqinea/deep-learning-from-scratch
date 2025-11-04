import numpy as np
import pytest
from chapters.ch05_ml_basics.api import as_one_hot, accuracy_from_logits

def test_as_one_hot_basic():
    labels = np.array([0, 2, 1])
    oh = as_one_hot(labels, num_classes=3)
    expected = np.array([[1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    assert oh.shape == (3,3)
    assert np.array_equal(oh, expected)

def test_as_one_hot_raises_on_bad_shape():
    with pytest.raises(ValueError):
        as_one_hot(np.array([[1,2]]), num_classes=3)

def test_accuracy_from_logits_with_ids():
    logits = np.array([[0.1, 0.9, -1.0],   # pred 1
                       [2.0, 1.0,  0.0],   # pred 0
                       [-3.0, 4.0, 5.0]])  # pred 2
    targets = np.array([1, 0, 2])
    acc = accuracy_from_logits(logits, targets)
    assert np.isclose(acc, 1.0)

def test_accuracy_from_logits_with_onehot():
    logits = np.array([[0.1, 0.9, -1.0],
                       [2.0, 1.0,  0.0],
                       [-3.0, 4.0, 5.0]])
    targets_oh = np.array([[0,1,0],
                           [1,0,0],
                           [0,0,1]])
    acc = accuracy_from_logits(logits, targets_oh)
    assert np.isclose(acc, 1.0)

def test_accuracy_from_logits_mismatch_batch_raises():
    logits = np.zeros((5, 3))
    targets = np.zeros((4,), dtype=int)
    with pytest.raises(ValueError):
        accuracy_from_logits(logits, targets)
