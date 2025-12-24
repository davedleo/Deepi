import numpy as np
from deepi.modules.activation import Activation 

class Tanh(Activation): 
    
    def __init__(self): 
        super().__init__("tanh")

    def forward(self, x: np.ndarray) -> np.ndarray: 
        return np.tanh(x)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        return dy * (1.0 - self.y ** 2.0)