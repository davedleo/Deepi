import numpy as np
from typing import Tuple
from deepi.modules.flow import Flow  

class Add(Flow): 
    
    def __init__(self): 
        super().__init__("add")

    def transform(self, xs: Tuple[np.ndarray, ...]) -> np.ndarray: 
        return sum(xs)
    
    def gradients(self, dy: np.ndarray) -> Tuple[np.ndarray, ...]:
        return tuple(dy for _ in self.x)