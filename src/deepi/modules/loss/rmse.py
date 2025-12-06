from typing import Optional
import numpy as np
from deepi.modules.loss import Loss

class RMSE(Loss):

    def __init__(self, reduction: Optional[str] = "mean", eps: float = 1e-8):
        super().__init__("rmse", reduction)
        self.eps = eps

    def transform(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        mse = ((y_hat - y) ** 2).mean(1)
        return np.sqrt(mse + self.eps)

    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = y_hat - y
        mse = np.mean(diff ** 2, axis=1, keepdims=True)
        return diff / (np.sqrt(mse + self.eps) * diff.shape[1])