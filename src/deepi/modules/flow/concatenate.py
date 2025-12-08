import numpy as np
from typing import Tuple
from deepi.modules.flow import Flow  

class Concatenate(Flow): 
    
    def __init__(self, axis: int = -1): 
        super().__init__("concatenate")
        self.axis = axis

    def transform(self, x: Tuple[np.ndarray, ...]) -> np.ndarray: 
        return 
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        return 