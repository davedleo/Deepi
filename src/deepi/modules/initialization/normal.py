from typing import Tuple 
import numpy as np 
from deepi.modules.initialization import Initializer 


class Normal(Initializer): 

    def __init__(
            self, 
            mean: float,
            std: float
    ): 
        super().__init__("normal")
        self.mean = mean 
        self.std = std 

    def rule(self, shape: Tuple[int, ...]) -> np.ndarray: 
        return np.random.uniform(self.mean, self.std, shape)