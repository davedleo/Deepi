import numpy as np
from deepi.modules.activation import Activation 

class LeakyReLU(Activation): 
    def __init__(self, negative_slope: float = 0.01): 
        super().__init__("leaky_relu")
        self.negative_slope = negative_slope

    def transform(self, x: np.ndarray) -> np.ndarray: 
        return np.where(x > 0.0, x, x * self.negative_slope)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = np.where(self.x > 0.0, 1.0, self.negative_slope) 
        return dx * dy