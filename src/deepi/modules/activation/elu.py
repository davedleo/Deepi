import numpy as np
from deepi.modules.activation import Activation

class ELU(Activation):
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("elu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, self.alpha * np.expm1(x))

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = np.where(self.x > 0.0, 1.0, self.alpha * np.exp(self.x))
        return dx * dy