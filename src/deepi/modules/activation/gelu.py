import numpy as np
from scipy.special import erf
from deepi.modules.activation import Activation

class GELU(Activation):
    
    def __init__(self, approximate: bool = True):
        super().__init__("gelu")
        self.approximate = approximate

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.approximate:
            x3 = x ** 3
            inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
            tanh_inner = np.tanh(inner)
            return 0.5 * x * (1.0 + tanh_inner)
        else:
            return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        x = self.x
        if self.approximate:
            x2 = x ** 2
            x3 = x ** 3
            inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
            tanh_inner = np.tanh(inner)
            sech2 = 1 - tanh_inner ** 2
            d_inner_dx = np.sqrt(2 / np.pi) * (1.0 + 3.0 * 0.044715 * x2)
            dx = 0.5 * (1.0 + tanh_inner + x * sech2 * d_inner_dx)
            return dx * dy
        else:
            erf_term = erf(x / np.sqrt(2))
            dx = 0.5 * (1.0 + erf_term) + (x / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
            return dx * dy