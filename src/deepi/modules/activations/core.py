from abc import abstractmethod

import numpy as np

from deepi.modules import Module


class Activation(Module):

    def __init__(
        self,
        _type: str
    ):
        super().__init__(f"activation.{_type}", False)


class ReLU(Activation):

    def __init__(self):
        super().__init__("relu")
        self.dx = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._is_training:
            self.dx = (x > 0.0).astype(float)
            y = x * self.dx 
        else: 
            y = np.maximum(x, 0.0)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx