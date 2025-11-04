from typing import Literal

def generalization_gap(train_value: float,
                       val_value: float,
                       metric: Literal["loss","accuracy"] = "loss") -> float:
    """
    For loss:  gap = val - train   (bigger gap -> worse generalization)
    For acc:   gap = train - val   (bigger gap -> worse generalization)
    """
    if metric == "loss":
        return float(val_value - train_value)
    if metric == "accuracy":
        return float(train_value - val_value)
    raise ValueError("metric must be 'loss' or 'accuracy'")

def is_overfitting_loss(train_loss: float,
                        val_loss: float,
                        gap_threshold: float = 0.1) -> bool:
    """
    Overfitting (loss view): val significantly higher than train.
    gap_threshold is absolute (e.g., 0.1 loss).
    """
    gap = val_loss - train_loss
    return (train_loss < val_loss) and (gap > gap_threshold)

def is_underfitting_loss(train_loss: float,
                         val_loss: float,
                         high_threshold: float = 1.0) -> bool:
    """
    Underfitting (loss view): both losses are high (model too weak).
    high_threshold is problem-dependent; default is didactic.
    """
    return (train_loss > high_threshold) and (val_loss > high_threshold)
