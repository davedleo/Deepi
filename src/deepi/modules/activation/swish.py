import numpy as np
from deepi.modules.activation import Activation

class Swish(Activation):

    def __init__(self, beta: float = 1.0):
        super().__init__("swish")
        self.beta = beta

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.beta * x
        sigmoid_z = np.empty_like(z, dtype=np.float64)
        mask = z >= 0
        sigmoid_z[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
        sigmoid_z[~mask] = np.exp(z[~mask]) / (1.0 + np.exp(z[~mask]))
        return x * sigmoid_z

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        z = self.beta * self.x
        sigmoid_z = np.empty_like(z, np.float64)
        mask = z >= 0
        sigmoid_z[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
        sigmoid_z[~mask] = np.exp(z[~mask]) / (1.0 + np.exp(z[~mask]))
        dy_swish = sigmoid_z + self.beta * self.x * sigmoid_z * (1.0 - sigmoid_z)
        return dy_swish * dy