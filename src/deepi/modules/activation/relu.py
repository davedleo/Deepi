import numpy as np
from deepi.modules.activation import Activation 

class ReLU(Activation): 
    
    def __init__(self): 
        super().__init__("relu")

    def forward(self, x: np.ndarray) -> np.ndarray: 
        return np.maximum(x, 0.0)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        return dy * (self.x > 0.0)