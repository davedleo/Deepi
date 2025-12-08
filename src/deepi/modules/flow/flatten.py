import numpy as np
from deepi.modules.flow import Flow 


class Flatten(Flow): 
    
    def __init__(self): 
        super().__init__("flatten")

    def transform(self, x: np.ndarray) -> np.ndarray: 
        return x.reshape(x.shape[0], -1)
    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dy_shape = dy.shape[:1] + self.x.shape[1:]
        return dy.reshape(dy_shape)