from typing import Optional
import numpy as np
from deepi.modules.loss import Loss

class Huber(Loss):

    def __init__(self, delta: float = 1.0, reduction: Optional[str] = "mean"):
        super().__init__("huber", reduction)
        self.delta = float(delta)

    def transform(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        loss_elem = abs_diff
        mask = abs_diff <= self.delta
        loss_elem = mask * (0.5 * diff**2) + (~mask) * (self.delta * (abs_diff - 0.5 * self.delta))
        batch_size = loss_elem.shape[0]
        return loss_elem.reshape(batch_size, -1).sum(axis=1)

    def gradients(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        mask = abs_diff <= self.delta
        return mask * diff + (~mask) * (self.delta * np.sign(diff))