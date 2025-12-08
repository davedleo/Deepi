import numpy as np
from typing import Tuple
from deepi.modules.flow import Flow  

class Concatenate(Flow): 
    
    def __init__(self, axis: int = -1): 
        super().__init__("concatenate")
        self.axis = axis

    def transform(self, xs: Tuple[np.ndarray, ...]) -> np.ndarray: 
        return np.concatenate(xs, axis=self.axis)
    
    def gradients(self, dy: np.ndarray) -> Tuple[np.ndarray, ...]:
        sizes = [x.shape[self.axis] for x in self.x]
        splits = np.split(dy, np.cumsum(sizes)[:-1], axis=self.axis)
        return tuple(splits)