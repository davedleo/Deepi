import numpy as np
from deepi.modules.activation import Activation

class ReLU6(Activation): 
    def __init__(self): 
        super().__init__("relu6")

    def transform(self, x: np.ndarray) -> np.ndarray: 
        return np.clip(x, 0.0, 6.0)  

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dx = np.where((self.x > 0.0) & (self.x < 6.0), 1.0, 0.0)
        return dx * dy