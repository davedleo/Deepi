import numpy as np
from deepi.modules.activation import Activation

class GLU(Activation):

    def __init__(self, axis: int = -1):
        super().__init__("glu")
        self.axis = axis

    def transform(self, x: np.ndarray) -> np.ndarray:
        a, b = np.split(x, 2, axis=self.axis)
        sigmoid_b = 1.0 / (1.0 + np.exp(-b))
        return a * sigmoid_b

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        a, b = np.split(self.x, 2, axis=self.axis)
        dy_a, dy_b = np.split(dy, 2, axis=self.axis)
        sigmoid_b = 1.0 / (1.0 + np.exp(-b))

        dx_a = dy_a * sigmoid_b
        dx_b = dy_b * a * sigmoid_b * (1.0 - sigmoid_b)
        return np.concatenate([dx_a, dx_b], axis=self.axis)