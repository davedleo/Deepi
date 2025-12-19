import numpy as np
from typing import Tuple
from deepi.modules.flow import Flow


class Input(Flow): 
    
    def __init__(self, in_shape: Tuple[int, ...]): 
        super().__init__("input")
        self.in_shape = in_shape

    def generate_sample(self) -> np.ndarray: 
        return np.empty((1,) + self.in_shape, dtype=np.float64)

    def transform(self, x: np.ndarray) -> np.ndarray: 
        return x
    
    def gradients(self, dy: np.ndarray):
        return