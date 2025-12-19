import numpy as np
from typing import Tuple
from deepi.modules.flow import Flow  

class Reshape(Flow): 
    
    def __init__(self, out_shape: Tuple[int, ...]): 
        super().__init__("reshape")
        self.out_shape = out_shape

    def forward(self, x: np.ndarray) -> np.ndarray: 
        out_shape = x.shape[:1] + self.out_shape
        return x.reshape(out_shape)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dy_shape = dy.shape[:1] + self.x.shape[1:]
        return dy.reshape(dy_shape)