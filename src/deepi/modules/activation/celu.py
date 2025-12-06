import numpy as np
from deepi.modules.activation import Activation

class CELU(Activation):
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("celu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, self.alpha * np.expm1(x / self.alpha))

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = np.where(self.x > 0.0, 1.0, np.exp(self.x / self.alpha))
        return dx * dy