import numpy as np
from deepi.modules.activation import Activation

class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        super().__init__("elu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        pos_mask = x > 0.0
        neg_mask = ~pos_mask
        y = x * pos_mask
        y[neg_mask] = self.alpha * np.expm1(x[neg_mask])
        return y

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        pos_mask = self.x > 0.0
        neg_mask = ~pos_mask
        dx = np.ones_like(self.x)
        dx[neg_mask] = self.alpha * np.exp(self.x[neg_mask])
        return dx * dy