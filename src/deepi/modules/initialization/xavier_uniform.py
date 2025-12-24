from typing import Tuple 
from math import sqrt 

import numpy as np 

from deepi.modules.initialization import Initializer 


class XavierUniform(Initializer): 

    def __init__(
            self, 
            gain: float = 1.0
    ): 
        super().__init__("xavier_uniform")
        self.gain = gain

    def init(self, shape: Tuple[int, ...]) -> np.ndarray: 
        fan_in = self.fan_in(shape)
        fan_out = self.fan_out(shape)
        r = self.gain * sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-r, r, shape)