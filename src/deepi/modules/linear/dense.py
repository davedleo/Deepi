import numpy as np
from deepi.modules.linear import Linear 

class Dense(Linear): 
    
    def __init__(
            self, 
            out_size: int,
            bias: bool = True
    ): 
        super().__init__("dense", bias, out_size)
        self.params["w"] = (out_size,)

    def transform(self, x: np.ndarray) -> np.ndarray: 
        y = x @ self.params["w"]

        if self._has_bias: 
            y += self.params["b"]

        return y

    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dy_dense = dy @ self.params["w"].T
        self.grads["w"] = self.x.T @ dy

        if self._has_bias: 
            self.grads["b"] = dy.sum(axis=0, keepdims=True)

        return dy_dense