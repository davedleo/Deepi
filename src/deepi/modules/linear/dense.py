import numpy as np
from typing import Dict, Tuple
from deepi.modules.linear import Linear 


class Dense(Linear): 
    
    def __init__(
            self, 
            out_size: int,
            bias: bool = True
    ): 
        super().__init__("dense", bias, out_size)
        self.params["w"] = (out_size,)

    def set_input(self, x: np.ndarray): 
        in_size = x.shape[1]
        out_size = self.params["w"][0]
        self.params["w"] = (in_size, out_size)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        y = x @ self.params["w"]

        if self._has_bias: 
            y += self.params["b"]

        return y

    def gradients(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        dy_dense = dy @ self.params["w"].T

        grads = {}
        grads["w"] = self.x.T @ dy
        if self._has_bias: 
            grads["b"] = dy.sum(axis=0, keepdims=True)

        return dy_dense, grads