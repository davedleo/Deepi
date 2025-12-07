import numpy as np
from deepi.modules.activation import Activation

class CELU(Activation):
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("celu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0.0
        pos = mask * x
        neg = (~mask) * self.alpha * np.expm1(x / self.alpha)
        return pos + neg

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        mask = self.x > 0.0
        dx_pos = mask * 1.0
        dx_neg = (~mask) * np.exp(self.x / self.alpha)
        return (dx_pos + dx_neg) * dy