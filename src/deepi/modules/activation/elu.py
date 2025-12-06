import numpy as np
from deepi.modules.activation import Activation

class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        super().__init__("elu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray:
        mask = x <= 0.0
        return np.where(mask, self.alpha * np.expm1(x), x)

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        mask = self.x <= 0.0
        dx = np.where(mask, self.alpha * np.exp(self.x), 1.0)
        return dx * dy