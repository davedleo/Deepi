import numpy as np
from deepi.modules.activation import Activation 

class LeakyReLU(Activation): 
    
    def __init__(self, alpha: float = 0.01): 
        super().__init__("leaky_relu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray: 
        return np.where(x > 0.0, x, x * self.alpha)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = np.where(self.x > 0.0, 1.0, self.alpha) 
        return dx * dy