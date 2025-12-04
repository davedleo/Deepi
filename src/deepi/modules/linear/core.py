from abc import abstractmethod 
from typing import Dict 

import numpy as np 

from deepi.modules import Module 


class Linear(Module):

    def __init__(
            self, 
            _type: str,
            out_features: int,
            bias: bool 
    ): 
        super().__init__(f"linear.{_type}", True)
        self._has_bias = bias 
        
        if self._has_bias: 
            self.params["b"] = np.zeros((1, out_features), dtype=float)

    @abstractmethod
    def set_input(self, x: np.ndarray): 
        raise NotImplementedError()


class Dense(Linear): 

    def __init__(
            self, 
            out_features: int, 
            bias: bool = True
    ): 
        super().__init__("dense", out_features, bias)
        self.params["w"] = (out_features,)

    def set_input(self, x: np.ndarray): 
        _, in_features = x.shape
        out_features = self.params["w"]
        self.params["w"] = (in_features, out_features)

    def forward(self, x: np.ndarray): 
        w = self.params["w"]
        y = x @ w 

        if self._is_training: 
            self.cache = w.T.copy()
            self.grads["w"] = lambda dy: x.T.copy() @ dy 
            if self._has_bias:
                b = self.params["b"]
                y += b
                self.grads["b"] = lambda dy: dy.sum(0, keepdims=True)

        elif self._has_bias: 
            b = self.params["b"]
            y += b

        return y 
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        wT = self.cache
        dy = dy @ wT
        return dy