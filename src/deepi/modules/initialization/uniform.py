from typing import Tuple 
import numpy as np 
from deepi.modules.initialization import Initializer 


class Uniform(Initializer): 

    def __init__(
            self, 
            low: float,
            high: float
    ): 
        super().__init__("uniform")
        self.low = low 
        self.high = high 

    def init(self, shape: Tuple[int, ...]) -> np.ndarray: 
        return np.random.uniform(self.low, self.high, shape)