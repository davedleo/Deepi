import numpy as np
from deepi.modules.activation import Activation 

class Tanh(Activation): 
    def __init__(self): 
        super().__init__("tanh")

    def transform(self, x: np.ndarray) -> np.ndarray: 
        return np.tanh(x)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = (1. - self.dy ** 2.0)
        return dx * dy