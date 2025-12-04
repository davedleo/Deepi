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

    @abstractmethod 
    def set_input(self, x: np.ndarray): 
        raise NotImplementedError


class Dense(Linear): 

    def __init__(
            self,
            layer_size: int, 
            bias: bool = True
    ):
        super().__init__("dense", layer_size, bias)
        self.params["w"] = (layer_size,)

    def set_input(self, x: np.ndarray):
        _, in_features = x.shape
        layer_size = self.params["w"][0]
        self.params["w"] = (in_features, layer_size)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        w = self.params["w"]
        y = x @ w

        if self._is_training:
            self.cache = w.T.copy() 
            self.grads["w"] = ...

            if self._has_bias: 
                b = self.params["b"]
                y += b
                self.grads["b"] = ...

        elif self._has_bias: 
            y += self.params["b"]

        return y 
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        wT = self.cache
        dy = dy @ wT
        return dy
    

class LowRank(Linear): 

    def __init__(
            self, 
            layer_size: int,
            rank: int = 1,
            bias: bool = True
    ): 
        super().__init__("low_rank", layer_size, bias)
        self.params["w1"] = (rank,)
        self.params["w2"] = (rank, layer_size)

    def set_input(self, x: np.ndarray): 
        _, in_features = x.shape 
        rank = self.params["w1"][0]
        self.params["w1"] = (in_features, rank)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        w1 = self.params["w1"]
        w2 = self.params["w2"]
        y = ... 

        if self._is_training: 
            self.cache = w2.T.copy(), w1.T.copy()
            if self._has_bias: 
                b = self.params["b"] 
                y += b 
                self.grads["b"] = np.ones_like(b)

        elif self._has_bias: 
            y += self.params["b"]
        
        return y
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        w2T, w1T = self.cache
        dy = dy @ w2T
        dy = dy @ w1T 
        return dy

        