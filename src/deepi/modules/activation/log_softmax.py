import numpy as np
from deepi.modules.activation import Activation

class LogSoftmax(Activation):

    def __init__(self, axis: int = -1):
        super().__init__("logsoftmax")
        self.axis = axis

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=self.axis, keepdims=True)
        shifted = x - x_max
        exp_shifted = np.exp(shifted)
        exp_shifted_sum = np.sum(exp_shifted, axis=self.axis, keepdims=True)
        return shifted - np.log(exp_shifted_sum)

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        exp_shifted = np.exp(self.x - np.max(self.x, axis=self.axis, keepdims=True))
        softmax = exp_shifted / np.sum(exp_shifted, axis=self.axis, keepdims=True)
        dot = np.sum(dy, axis=self.axis, keepdims=True)
        return dy - softmax * dot