import numpy as np
from deepi.modules import Module


class Activation(Module):

    def __init__(
        self,
        _type: str
    ):
        super().__init__(f"activation.{_type}", False)


class LeakyReLU(Activation):

    def __init__(self, negative_slope: float = 0.01):
        super().__init__("leaky_relu")
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        y = x * mask + x * self.negative_slope * (~mask)
        if self._is_training:
            self.dx = mask + (~mask) * self.negative_slope
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx


class ReLU(Activation):

    def __init__(self):
        super().__init__("relu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        y = x * mask
        if self._is_training:
            self.dx = mask
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.where(self.dx, dy, 0.0)


class ReLU6(Activation):

    def __init__(self):
        super().__init__("relu6")

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = (x > 0.0) & (x < 6.0)
        y = mask * x + (~mask) * ((x > 6.0) * 6.0)
        if self._is_training:
            self.dx = mask
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.where(self.dx, dy, 0.0)