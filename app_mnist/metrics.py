import numpy as np
from chapters.ch05_ml_basics.api import accuracy_from_logits

def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(accuracy_from_logits(logits, targets))
