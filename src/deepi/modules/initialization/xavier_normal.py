from typing import Tuple 
from math import sqrt 

import numpy as np 

from deepi.modules.initialization import Initializer 


class XavierNormal(Initializer): 

    def __init__(
            self, 
            gain: float = 1.0
    ): 
        super().__init__("xavier_normal")
        self.gain = gain

    def rule(self, shape: Tuple[int, ...]) -> np.ndarray: 
        fan_in = self.fan_in(shape)
        fan_out = self.fan_out(shape)
        std = self.gain * sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, std, shape)