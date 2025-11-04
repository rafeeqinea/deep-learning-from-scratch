import numpy as np
from chapters.ch05_ml_basics.api import (
    generalization_gap, is_overfitting_loss, is_underfitting_loss
)

def test_generalization_gap_loss_and_accuracy():
    # loss: val - train
    assert np.isclose(generalization_gap(0.4, 0.6, metric="loss"), 0.2)
    # acc: train - val
    assert np.isclose(generalization_gap(0.92, 0.85, metric="accuracy"), 0.07)

def test_is_overfitting_loss():
    # val significantly higher than train (gap > threshold)
    assert is_overfitting_loss(0.2, 0.45, gap_threshold=0.1) is True
    # borderline or inverted
    assert is_overfitting_loss(0.2, 0.28, gap_threshold=0.1) is False
    assert is_overfitting_loss(0.3, 0.25, gap_threshold=0.1) is False

def test_is_underfitting_loss():
    # both high -> underfitting
    assert is_underfitting_loss(1.5, 1.7, high_threshold=1.0) is True
    # low losses -> not underfitting
    assert is_underfitting_loss(0.5, 0.6, high_threshold=1.0) is False
