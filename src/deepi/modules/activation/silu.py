import numpy as np
from deepi.modules.activation import Activation

class SiLU(Activation):

    def __init__(self):
        super().__init__("silu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = np.empty_like(x, np.float64)
        mask = x >= 0
        sigmoid_x[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        sigmoid_x[~mask] = np.exp(x[~mask]) / (1.0 + np.exp(x[~mask]))
        return x * sigmoid_x

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        x = self.x
        sigmoid_x = np.empty_like(x, np.float64)
        mask = x >= 0
        sigmoid_x[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        sigmoid_x[~mask] = np.exp(x[~mask]) / (1.0 + np.exp(x[~mask]))
        dy_silu = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
        return dy_silu * dy