import numpy as np
from deepi.modules.activation import Activation 

class LeakyReLU(Activation): 
    
    def __init__(self, alpha: float = 0.01): 
        super().__init__("leaky_relu")
        self.alpha = alpha

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[x < 0.0] *= self.alpha
        return y

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dy = dy.copy()
        dy[self.x <= 0.0] *= self.alpha
        return dy