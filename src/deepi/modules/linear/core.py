from abc import abstractmethod 
from typing import Dict 

import numpy as np

from deepi.modules import Module 


class Linear(Module): 

    def __init__(
            self, 
            _type: str,
            layer_size: int,
            bias: bool 
    ): 
        super().__init__(f"linear.{_type}", True) 
        self._has_bias = bias 

        if self._has_bias: 
            self.params["b"] = np.zeros((1, layer_size), dtype=float) 


class Dense(Linear): 

    def __init__(
            self,
            layer_size: int, 
            bias: bool = True
    ):
        super().__init__("dense", layer_size, bias)
        self.params["w"] = (layer_size,)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        w = self.params["w"]
        y = x @ w

        if self._is_training:
            self.cache = w.T.copy() 
            self.grads["w"] = x.T 

            if self._has_bias: 
                b = self.params["b"]
                y += b
                self.grads["b"] = np.ones_like(b)

        else: 
            y += self.params["b"]

        return y 
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        wT = self.cache
        return dy @ wT
        