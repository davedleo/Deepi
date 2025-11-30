import numpy as np
from scipy.special import erf

from deepi.modules import Module


class Activation(Module):

    def __init__(
        self,
        _type: str
    ):
        super().__init__(f"activation.{_type}", False)


class CELU(Activation):

    def __init__(self, alpha: float = 1.0):
        super().__init__("celu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        neg = (~mask)
        y = x * mask + (self.alpha * (np.exp((x * neg) / self.alpha) - 1.0)) * neg
        if self._is_training:
            self.dx = mask + neg * np.exp((x * neg) / self.alpha)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx


class ELU(Activation):

    def __init__(self, alpha: float = 1.0):
        super().__init__("elu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        neg = (~mask)
        y = x * mask + (self.alpha * (np.exp(x * neg) - 1.0)) * neg
        if self._is_training:
            self.dx = mask + neg * (self.alpha * np.exp(x * neg))
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx


class GLU(Activation):

    def __init__(self, axis: int = -1):
        super().__init__("glu")
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        a, b = np.split(x, 2, axis=self.axis)
        b_sig = 1.0 / (1.0 + np.exp(-b))
        y = a * b_sig
        if self._is_training:
            self.dx = (a, b_sig)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        a, b_sig = self.dx
        da = dy * b_sig
        db = dy * a * b_sig * (1.0 - b_sig)
        return np.concatenate([da, db], axis=self.axis)


class GELU(Activation):

    def __init__(self, approximate: bool = True):
        super().__init__("gelu")
        self.approximate = approximate
        if self.approximate: 
            self.c = 0.7978845608028654

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.approximate:
            # c = sqrt(2/pi)
            y = 0.5 * x * (1.0 + np.tanh(self.c * (x + 0.044715 * (x ** 3))))
        else:
            y = 0.5 * x * (1.0 + erf(x / np.sqrt(2)))
        if self._is_training:
            self.dx = x
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.approximate:
            t = self.c * (self.dx + 0.044715 * self.dx**3)
            th = np.tanh(t)
            sech2 = 1.0 - th**2
            # derivative of approximate GELU
            dx = 0.5 * (1.0 + th + self.dx * sech2 * self.c * (1.0 + 3 * 0.044715 * self.dx**2))
        else:
            # exact derivative
            dx = 0.5 * (1.0 + erf(self.dx / np.sqrt(2))) + (self.dx * np.exp(-self.dx**2 / 2)) / np.sqrt(2 * np.pi)

        return dy * dx


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


class SELU(Activation):

    def __init__(self):
        super().__init__("selu")
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        neg = (~mask)
        y = self.scale * (x * mask + (self.alpha * (np.exp(x * neg) - 1.0)) * neg)
        if self._is_training:
            self.dx = self.scale * (mask + neg * (self.alpha * np.exp(x * neg)))
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = 1.0 / (1.0 + np.exp(-x))
        if self._is_training:
            self.dx = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = dy * self.dx * (1.0 - self.dx)
        return dx


class SiLU(Activation):

    def __init__(self):
        super().__init__("silu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-x))
        y = x * sig
        if self._is_training:
            self.dx = (sig, x)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        sig, x = self.dx
        dx = dy * (sig + x * sig * (1.0 - sig))
        return dx


class Swish(Activation):

    def __init__(self):
        super().__init__("swish")

    def forward(self, x: np.ndarray) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-x))
        y = x * sig
        if self._is_training:
            self.dx = sig + x * sig * (1.0 - sig)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx


class Tanh(Activation):

    def __init__(self):
        super().__init__("tanh")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        if self._is_training:
            self.dx = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = dy * (1.0 - self.dx**2)
        return dx