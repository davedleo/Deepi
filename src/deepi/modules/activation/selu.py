import numpy as np
from deepi.modules.activation import Activation

class SELU(Activation):

    def __init__(self):
        super().__init__("selu")
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, self.scale * x, self.scale * self.alpha * np.expm1(x))

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = np.where(x > 0.0, self.scale, self.scale * self.alpha * np.exp(self.x))
        return dx * dy