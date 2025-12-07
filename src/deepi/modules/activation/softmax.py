import numpy as np
from deepi.modules.activation import Activation

class Softmax(Activation):

    def __init__(self, axis: int = -1):
        super().__init__("softmax")
        self.axis = axis

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dot = np.sum(self.y * dy, axis=self.axis, keepdims=True)
        return self.y * (dy - dot)