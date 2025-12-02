import numpy as np
from scipy.special import erf

from deepi.modules import Module


class Activation(Module):

    def __init__(
        self,
        _type: str
    ):
        super().__init__(f"activation.{_type}", False)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return self.dx * dy


class CELU(Activation):

    def __init__(self, alpha: float = 1.0):
        super().__init__("celu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0.0, x, self.alpha * np.expm1(x / self.alpha))
        if self._is_training:
            self.dx = np.where(x > 0.0, 1.0, np.exp(x / self.alpha))

        return y


class ELU(Activation):

    def __init__(self, alpha: float = 1.0):
        super().__init__("elu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0.0, x, self.alpha * np.expm1(x))

        if self._is_training:
            self.dx = np.where(x > 0.0, 1.0, self.alpha * np.exp(x))

        return y
    

class GELU(Activation):

    def __init__(self, approximate: bool = True):
        super().__init__("gelu")
        self.approximate = approximate

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.approximate:
            # Approximation using tanh: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
            x3 = x ** 3
            inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
            tanh_inner = np.tanh(inner)
            y = 0.5 * x * (1.0 + tanh_inner)

            if self._is_training:
                sech2 = 1 - tanh_inner ** 2
                self.dx = 0.5 * (1.0 + tanh_inner + x * sech2 * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2))
        else:
            y = 0.5 * x * (1 + erf(x / np.sqrt(2)))
            if self._is_training:
                self.dx = 0.5 * (1.0 + erf(x / np.sqrt(2))) + (x / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)

        return y


class GLU(Activation):

    def __init__(self, axis: int = -1):
        super().__init__("glu")
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        a, b = np.split(x, 2, axis=self.axis)
        sigmoid_b = 1.0 / (1.0 + np.exp(-b))
        y = a * sigmoid_b

        if self._is_training:
            dx_a = sigmoid_b
            dx_b = a * sigmoid_b * (1.0 - sigmoid_b)
            self.dx = np.concatenate([dx_a, dx_b], axis=self.axis)

        return y


class LeakyReLU(Activation):

    def __init__(self, alpha: float = 0.01):
        super().__init__("leaky_relu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0.0, x, self.alpha * x)

        if self._is_training:
            self.dx = np.where(x > 0.0, 1.0, self.alpha)

        return y


class ReLU(Activation):

    def __init__(self):
        super().__init__("relu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.maximum(x, 0.0)

        if self._is_training:
            self.dx = (x > 0.0).astype(float)

        return y


class ReLU6(Activation):

    def __init__(self):
        super().__init__("relu6")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.clip(x, 0.0, 6.0)

        if self._is_training:
            self.dx = ((x > 0.0) & (x < 6.0)).astype(float)

        return y


class SELU(Activation):

    def __init__(self):
        super().__init__("selu")
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0.0, self.scale * x, self.scale * self.alpha * np.expm1(x))

        if self._is_training:
            self.dx = np.where(x > 0.0, self.scale, self.scale * self.alpha * np.exp(x))

        return y


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        mask = x >= 0
        y[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        y[~mask] = np.exp(x[~mask]) / (1.0 + np.exp(x[~mask]))

        if self._is_training:
            self.dx = y * (1.0 - y)

        return y


class SiLU(Activation):

    def __init__(self):
        super().__init__("silu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = np.empty_like(x)
        mask = x >= 0
        sigmoid_x[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        sigmoid_x[~mask] = np.exp(x[~mask]) / (1.0 + np.exp(x[~mask]))
        y = x * sigmoid_x

        if self._is_training:
            self.dx = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))

        return y


class Swish(Activation):

    def __init__(self, beta: float = 1.0):
        super().__init__("swish")
        self.beta = beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.beta * x
        sigmoid_z = np.empty_like(z)
        mask = z >= 0
        sigmoid_z[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
        sigmoid_z[~mask] = np.exp(z[~mask]) / (1.0 + np.exp(z[~mask]))
        y = x * sigmoid_z

        if self._is_training:
            self.dx = sigmoid_z + self.beta * x * sigmoid_z * (1.0 - sigmoid_z)

        return y


class Tanh(Activation):

    def __init__(self):
        super().__init__("tanh")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)

        if self._is_training:
            self.dx = 1.0 - y ** 2

        return y


class Softmax(Activation):

    def __init__(self, axis: int = -1):
        super().__init__("softmax")
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        y = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        if self._is_training:
            self.dx = y

        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dot = np.sum(self.dx * dy, axis=self.axis, keepdims=True)
        return self.dx * (dy - dot)
