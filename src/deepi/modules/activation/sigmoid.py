import numpy as np
from deepi.modules.activation import Activation

class Sigmoid(Activation): 
    
    def __init__(self): 
        super().__init__("sigmoid")

    def forward(self, x: np.ndarray) -> np.ndarray: 
        pos_mask = x >= 0.0
        neg_mask = ~pos_mask
        y = np.empty_like(x, dtype=np.float64)
        y[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        y[neg_mask] = np.exp(x[neg_mask]) / (1.0 + np.exp(x[neg_mask]))
        return y

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.y * (1.0 - self.y)