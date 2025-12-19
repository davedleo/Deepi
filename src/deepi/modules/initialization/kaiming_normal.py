from typing import Tuple 
from math import sqrt 

import numpy as np 

from deepi.modules.initialization import Initializer 


class KaimingNormal(Initializer): 

    def __init__(
            self, 
            gain: float = 1.0,
            fan_mode: str = "in"
    ): 
        super().__init__("kaiming_normal")
        self.gain = gain 
        
        if fan_mode not in {"in", "out", "avg"}: 
            raise ValueError()
        
        self.fan_mode = fan_mode

    def init(self, shape: Tuple[int, ...]) -> np.ndarray: 
        if self.fan_mode == "in": 
            fan = self.fan_in(shape)
        elif self.fan_mode == "out": 
            fan = self.fan_out(shape)
        else: # avg 
            fan_in = self.fan_in(shape)
            fan_out = self.fan_out(shape)
            fan = 0.5 * (fan_in + fan_out)
        
        std = self.gain / sqrt(fan)

        return np.random.normal(0.0, std, shape)