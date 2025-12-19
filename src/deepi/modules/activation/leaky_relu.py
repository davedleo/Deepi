import numpy as np
from deepi.modules.activation import Activation

class LeakyReLU(Activation): 
    
    def __init__(self, alpha: float = 0.01): 
        super().__init__("leaky_relu")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x >= 0.0
        return mask * x + (~mask) * self.alpha * x

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        mask = self.x >= 0.0
        return mask * dy + (~mask) * self.alpha * dy