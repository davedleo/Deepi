import numpy as np
from deepi.modules.activation import Activation

class ELU(Activation):
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("elu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray:
        mask = x <= 0.0
        y = x.copy()
        y[mask] = self.alpha * np.expm1(x[mask])
        return y

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        mask = self.x <= 0.0
        dy_elu = np.ones_like(dy)
        dy_elu[mask] = self.alpha * np.exp(self.x[mask])
        return dy_elu * dy